#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.quantization as tq

from sentence_infer import SentenceCNN


LAYERS = ["conv1", "conv2", "conv3", "fc1", "fc2"]


def round_away(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int32)


def clip_i8(x):
    return np.clip(x, -127, 127).astype(np.int8)


def requantize(acc, scale, zp=0, relu=False):
    q = clip_i8(round_away(acc.astype(np.float32) * np.float32(scale)) + int(zp))
    if relu:
        q = np.maximum(q, 0).astype(np.int8)
    return q


def parse_input_header(path):
    text = Path(path).read_text()
    values = [int(v) for v in re.findall(r"\{(-?\d+)\}", text)]
    if len(values) != 384:
        raise RuntimeError(f"Expected 384 input values in {path}, got {len(values)}")
    valid = "valid_embedding = true" in text
    return np.asarray(values, dtype=np.int8), valid


def build_quantized_shell(weight_path):
    model = SentenceCNN()
    model.qconfig = tq.QConfig(
        activation=tq.HistogramObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
        weight=tq.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
    )
    tq.prepare(model, inplace=True)
    tq.convert(model, inplace=True)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    state = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def extract_params(model):
    modules = dict(model.named_modules())
    params = {}
    prev_scale = 1.0
    for name in LAYERS:
        mod = modules[name]
        wq = mod.weight()
        if wq.qscheme() not in (torch.per_tensor_symmetric, torch.per_tensor_affine):
            raise RuntimeError(f"{name} is {wq.qscheme()}, expected per-tensor")
        w_scale = float(wq.q_scale())
        out_scale = float(getattr(mod, "scale", 1.0))
        bias_fp = mod.bias().detach().cpu().numpy() if mod.bias() is not None else None
        bias_i32 = None if bias_fp is None else np.round(bias_fp / (prev_scale * w_scale)).astype(np.int32)
        params[name] = {
            "w": wq.int_repr().cpu().numpy().astype(np.int8),
            "bias": bias_i32,
            "requant": (prev_scale * w_scale) / out_scale,
            "out_scale": out_scale,
        }
        prev_scale = out_scale
    return params


def conv1d_i8(x_lc, w_oik, bias, scale, padding, relu=True):
    length, in_ch = x_lc.shape
    out_ch, w_in_ch, kernel = w_oik.shape
    if in_ch != w_in_ch:
        raise RuntimeError(f"Conv input channels mismatch: {in_ch} vs {w_in_ch}")
    out = np.zeros((length, out_ch), dtype=np.int8)
    for pos in range(length):
        for oc in range(out_ch):
            acc = int(bias[oc]) if bias is not None else 0
            for k in range(kernel):
                src = pos + k - padding
                if 0 <= src < length:
                    acc += int(np.dot(x_lc[src, :].astype(np.int32), w_oik[oc, :, k].astype(np.int32)))
            out[pos, oc] = requantize(np.asarray(acc, dtype=np.int32), scale, relu=relu)
    return out


def linear_i8(x_i, w_oi, bias, scale, relu):
    acc = x_i.astype(np.int32) @ w_oi.astype(np.int32).T
    if bias is not None:
        acc = acc + bias.astype(np.int32)
    return requantize(acc, scale, relu=relu)


def staged_pool_proxy(conv3_lc):
    x = conv3_lc.reshape(16, 24, 256)
    for out_h, out_w in [(8, 12), (4, 6), (2, 3)]:
        y = np.zeros((out_h, out_w, 256), dtype=np.int8)
        for h in range(out_h):
            for w in range(out_w):
                y[h, w, :] = x[h * 2:h * 2 + 2, w * 2:w * 2 + 2, :].max(axis=(0, 1))
        x = y
    return x[0, 0, :]


def run_path(params, emb, pool_mode):
    x = emb.reshape(384, 1)
    c1 = conv1d_i8(x, params["conv1"]["w"], params["conv1"]["bias"], params["conv1"]["requant"], padding=2)
    c2 = conv1d_i8(c1, params["conv2"]["w"], params["conv2"]["bias"], params["conv2"]["requant"], padding=2)
    c3 = conv1d_i8(c2, params["conv3"]["w"], params["conv3"]["bias"], params["conv3"]["requant"], padding=1)
    if pool_mode == "exact":
        pooled = c3.max(axis=0)
    elif pool_mode == "staged_proxy":
        pooled = staged_pool_proxy(c3)
    else:
        raise RuntimeError(f"Unsupported pool_mode: {pool_mode}")
    fc1 = linear_i8(pooled, params["fc1"]["w"], params["fc1"]["bias"], params["fc1"]["requant"], relu=True)
    fc2 = linear_i8(fc1, params["fc2"]["w"], params["fc2"]["bias"], params["fc2"]["requant"], relu=False)
    q = int(fc2.reshape(-1)[0])
    logit = q * params["fc2"]["out_scale"]
    prob = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, logit))))
    return q, logit, prob


def main():
    parser = argparse.ArgumentParser(description="Check the Gemmini per-tensor integer path on CPU")
    parser.add_argument("--weights", default="weights/sentence_cnn_int8_gemmini_per_tensor.pth")
    parser.add_argument("--input-header", default="../../../bareMetalC/random_embedding_gemmini.h")
    args = parser.parse_args()

    emb, valid = parse_input_header(args.input_header)
    model = build_quantized_shell(args.weights)
    params = extract_params(model)

    print(f"input_header={args.input_header}")
    print(f"valid_embedding={valid}")
    for name in LAYERS:
        p = params[name]
        print(f"{name},requant={p['requant']:.8f},out_scale={p['out_scale']:.8f},w_shape={list(p['w'].shape)}")

    for mode in ["exact", "staged_proxy"]:
        q, logit, prob = run_path(params, emb, mode)
        pred = prob >= 0.5
        print(f"pool={mode},logit_q={q},logit={logit:.6f},prob={prob:.6f},pred={int(pred)},correct={pred == valid}")


if __name__ == "__main__":
    main()
