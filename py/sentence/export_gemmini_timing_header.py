#!/usr/bin/env python3
"""
Export a Gemmini-oriented timing header for SentenceCNN.

This is intentionally a performance-oriented export, not the accuracy-matching
RVV/per-channel export:
- conv weights are exported as HWIO for Gemmini native tiled_conv_auto
- Conv1d kernels are embedded in the center row of square 2D kernels
- requantization scales are per-tensor scalar values for Gemmini store scaling
- a 1x1 identity HWIO kernel is emitted for staged Gemmini pooling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from sentence_infer import SentenceCNN


ORDERED_LAYERS = ["conv1", "conv2", "conv3", "fc1", "fc2"]


def build_quantized_model(weight_path: str, device: torch.device):
    torch.backends.quantized.engine = "fbgemm"

    model = SentenceCNN()
    model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.HistogramObserver.with_args(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)

    sd = torch.load(weight_path, map_location=device)
    state = sd["model_state_dict"] if isinstance(sd, dict) and "model_state_dict" in sd else sd
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def representative_scalar_scale(scales: np.ndarray, mode: str) -> float:
    if mode == "first":
        return float(scales[0])
    if mode == "mean":
        return float(np.mean(scales))
    if mode == "max":
        return float(np.max(scales))
    raise ValueError(f"unsupported scale mode: {mode}")


def conv1d_weight_to_square_hwio(w_int8: np.ndarray) -> np.ndarray:
    # PyTorch Conv1d quant weight: O, I, K. Gemmini conv expects H, W, I, O.
    if w_int8.ndim != 3:
        raise ValueError(f"expected Conv1d weight ndim=3, got {w_int8.shape}")
    out_ch, in_ch, kernel = w_int8.shape
    hwio = np.zeros((kernel, kernel, in_ch, out_ch), dtype=np.int8)
    center = kernel // 2
    for o in range(out_ch):
        for i in range(in_ch):
            for k in range(kernel):
                hwio[center, k, i, o] = w_int8[o, i, k]
    return hwio


def linear_weight_to_io(w_int8: np.ndarray) -> np.ndarray:
    # Gemmini tiled_matmul uses A[I,K] * B[K,J], so transpose Linear O,I to I,O.
    if w_int8.ndim != 2:
        raise ValueError(f"expected Linear weight ndim=2, got {w_int8.shape}")
    return np.ascontiguousarray(w_int8.T)


def write_nested_array(f, ctype: str, name: str, arr: np.ndarray):
    dims = "".join(f"[{d}]" for d in arr.shape)
    f.write(f"{ctype} {name}{dims} = ")

    def rec(x, level=0):
        if not isinstance(x, np.ndarray):
            return str(int(x))
        if x.ndim == 1:
            return "{" + ",".join(str(int(v)) for v in x) + "}"
        indent = "    " * level
        child_indent = "    " * (level + 1)
        return "{\n" + ",\n".join(child_indent + rec(v, level + 1) for v in x) + "\n" + indent + "}"

    f.write(rec(arr))
    f.write(";\n\n")


def write_float_macros(f, scales: dict[str, float]):
    for name, value in scales.items():
        macro = f"{name.upper()}_REQUANT_SCALE"
        f.write(f"#ifndef {macro}\n")
        f.write(f"#define {macro} {value:.8f}f\n")
        f.write("#endif\n")
    f.write("\n")


def export_header(weight_path: str, out_path: Path, scale_mode: str):
    model = build_quantized_model(weight_path, torch.device("cpu"))
    modules = dict(model.named_modules())

    prev_scale = 1.0
    scalar_requant = {}
    arrays: dict[str, np.ndarray] = {}
    biases: dict[str, np.ndarray] = {}

    for name in ORDERED_LAYERS:
        module = modules[name]
        w_q = module.weight()
        w_int8 = w_q.int_repr().cpu().numpy().astype(np.int8)

        qscheme = w_q.qscheme()
        if qscheme in (torch.per_tensor_symmetric, torch.per_tensor_affine):
            weight_scales = np.array([float(w_q.q_scale())], dtype=np.float32)
        elif qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
            weight_scales = w_q.q_per_channel_scales().cpu().numpy().astype(np.float32)
        else:
            raise RuntimeError(f"unsupported qscheme for {name}: {qscheme}")

        out_scale = float(getattr(module, "scale", 1.0))
        per_channel_m = (prev_scale * weight_scales) / out_scale
        scalar_requant[name] = representative_scalar_scale(per_channel_m, scale_mode)

        if name.startswith("conv"):
            arrays[f"{name}_weight_hwio"] = conv1d_weight_to_square_hwio(w_int8)
        else:
            arrays[f"{name}_weights_mat"] = linear_weight_to_io(w_int8)

        if hasattr(module, "bias") and module.bias() is not None:
            b_fp32 = module.bias().detach().cpu().numpy()
            denom = prev_scale * weight_scales
            if denom.size == 1:
                b_int32 = np.round(b_fp32 / denom[0]).astype(np.int32)
            else:
                b_int32 = np.round(b_fp32 / denom).astype(np.int32)
            biases[f"{name}_bias"] = b_int32

        prev_scale = out_scale

    identity = np.eye(256, dtype=np.int8).reshape(1, 1, 256, 256)

    with out_path.open("w") as f:
        guard = "WEIGHTS_GEMMINI_TIMING_H"
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("// Auto-generated Gemmini timing header.\n")
        f.write("// This header prioritizes Gemmini dataflow timing over accuracy-equivalent quantization.\n")
        f.write(f"// scalar requant scale mode: {scale_mode}\n\n")
        write_float_macros(f, scalar_requant)
        for name, arr in arrays.items():
            write_nested_array(f, "elem_t", name, arr)
        write_nested_array(f, "elem_t", "pool_identity_1x1_hwio", identity)
        for name, arr in biases.items():
            write_nested_array(f, "acc_t", name, arr)
        f.write(f"#endif // {guard}\n")


def main():
    parser = argparse.ArgumentParser(description="Export Gemmini-native timing weights for SentenceCNN")
    parser.add_argument("--weights", default="weights/sentence_cnn_int8.pth")
    parser.add_argument("--out", default="weights_gemmini_timing.h")
    parser.add_argument("--scale-mode", choices=["first", "mean", "max"], default="first")
    args = parser.parse_args()

    export_header(args.weights, Path(args.out), args.scale_mode)
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
