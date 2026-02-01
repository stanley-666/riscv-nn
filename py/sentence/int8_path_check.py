"""
Manual int8 forward that mirrors the C path:
- Uses quantized conv/linear modules from the calibrated checkpoint.
- Reimplements AdaptiveMaxPool1d in int8 (per-tensor) to avoid FP fallback.
Run to get logit_q/logit_fp that should match test_q.c (weights_q.h).
"""

import argparse
import torch
import pandas as pd
import numpy as np

from sentence_infer import SentenceCNN


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


def q_adaptive_max_pool1d(qx: torch.Tensor, output_size: int = 1) -> torch.Tensor:
    """
    AdaptiveMaxPool1d implemented by:
    1) dequantize to fp32
    2) pool in fp32 (output_size=1)
    3) round half-away-from-zero back to int domain
    4) requantize with *same* scale/zp so the following quantized FC sees quint8.

    This matches the user's request: pool在浮點，但結果是整數，然後強制轉回 int8。
    """
    assert qx.ndim == 3, "expected (N, C, L)"
    if output_size != 1:
        raise ValueError("Only output_size=1 supported here")

    # fp32 pool
    xf = torch.nn.functional.adaptive_max_pool1d(qx.dequantize(), output_size)  # (N,C,1)

    # round half away from zero to mimic C side
    xf_rounded = torch.sign(xf) * torch.floor(torch.abs(xf) + 0.5)

    # requantize using the same activation scale/zp
    return torch.quantize_per_tensor(
        xf_rounded, scale=qx.q_scale(), zero_point=qx.q_zero_point(), dtype=torch.quint8
    )


def forward_int8(model, emb_int8: np.ndarray, transpose_wc_to_cw: bool = False, stop_after_conv3: bool = False):
    # Input assumed already int8 with scale=1, zp=0 (matches weights_q export).
    x_q = torch.quantize_per_tensor(
        torch.tensor(emb_int8, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        scale=1.0,
        zero_point=0,
        dtype=torch.quint8,
    )

    x = model.conv1(x_q)
    print("[py] conv1 dst[:8]:", x.int_repr().view(-1)[:8].tolist())
    x = model.conv2(x)
    print("[py] conv2 dst[:8]:", x.int_repr().view(-1)[:8].tolist())
    x = model.conv3(x)
    print("[py] conv3 dst[:8]:", x.int_repr().view(-1)[:8].tolist())

    if stop_after_conv3:
        return x

    if transpose_wc_to_cw:
        # PyTorch conv1d 輸出已是 (N, C, L)。C 端轉置是為了把 WC 轉回 CW。
        # 這裡保持不變（等效於已經在 CW），避免打亂維度導致 FC K 維錯誤。
        pass

    x = q_adaptive_max_pool1d(x, output_size=1)
    print("[py] pool dst[:8]:", x.int_repr().view(-1)[:8].tolist())

    x = x.squeeze(-1)  # (N, C)
    x = model.fc1(x)
    print("[py] fc1 dst[:8]:", x.int_repr().view(-1)[:8].tolist())

    x = model.fc2(x)   # quantized output
    print("[py] fc2 dst[:8]:", x.int_repr().view(-1)[:8].tolist())
    return x


def main():
    parser = argparse.ArgumentParser(description="INT8 path check (C-aligned) using quantized modules")
    parser.add_argument("--csv", type=str, default="embeddings/test_rest_int8.csv")
    parser.add_argument("--row", type=int, default=2658)
    parser.add_argument("--weights", type=str, default="weights/sentence_cnn_int8.pth")
    parser.add_argument("--transpose", action="store_true", help="mimic C-side TRANSPOSE_WC_TO_CW after conv3")
    parser.add_argument("--dump_conv3", action="store_true", help="stop after conv3 and dump a few logits")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = build_quantized_model(args.weights, device)

    df = pd.read_csv(args.csv)
    idx = max(0, min(args.row, len(df) - 1))
    emb = df[[f"q{i}" for i in range(384)]].iloc[idx].to_numpy(dtype="int8")
    label = int(df["label"].iloc[idx]) if "label" in df.columns else None

    with torch.no_grad():
        yq = forward_int8(model, emb, transpose_wc_to_cw=args.transpose, stop_after_conv3=args.dump_conv3)

    if args.dump_conv3:
        # Dump first few positions of conv3 output (channel 0) without transpose
        q_vals = yq.int_repr().squeeze(0)  # (C, W)
        scale = float(yq.q_scale())
        zp = int(yq.q_zero_point())
        print(f"row={idx}, label={label}, conv3 shape={list(q_vals.shape)}, scale={scale:.8f}, zp={zp}")
        ch0 = q_vals[0]
        print("conv3[0][:16] qint8:", ch0[:16].tolist())
        fp0 = (ch0[:16].float() - zp) * scale
        print("conv3[0][:16] fp   :", [float(v) for v in fp0])
        return

    q = int(yq.int_repr().item())
    scale = float(yq.q_scale())
    zp = int(yq.q_zero_point())
    logit_fp = (q - zp) * scale

    # C uses roundf (half away from zero) + clip to int8
    def round_half_away_from_zero(x: torch.Tensor):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    logit_fp_t = torch.tensor(logit_fp, dtype=torch.float32)
    q_c = round_half_away_from_zero(logit_fp_t / scale) + zp
    q_c = torch.clamp(q_c, -127, 127).to(torch.int32)
    logit_fp_c = (q_c - zp) * scale

    prob_bankers = 1.0 / (1.0 + np.exp(-logit_fp))
    prob_c = 1.0 / (1.0 + np.exp(-float(logit_fp_c)))

    print(f"row={idx}, label={label}")
    print(f"[PyTorch quant kernel] logit_q={q}, scale={scale:.8f}, zp={zp}, logit_fp={logit_fp:.6f}, prob={prob_bankers:.6f}")
    print(f"[C rounding emu     ] logit_q={int(q_c)}, logit_fp={float(logit_fp_c):.6f}, prob={prob_c:.6f}")


if __name__ == "__main__":
    main()
