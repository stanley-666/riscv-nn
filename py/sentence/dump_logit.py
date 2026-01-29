import argparse
import torch
import torch.quantization as tq
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math

from sentence_infer import SentenceCNN


def build_quantized_shell():
    model = SentenceCNN()
    model.qconfig = tq.QConfig(
        activation=tq.HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    )
    tq.prepare(model, inplace=True)
    tq.convert(model, inplace=True)
    return model


def dequantize_to_float_model(q_state_dict: dict) -> SentenceCNN:
    """Create a fresh float model and copy dequantized weights/biases from a quantized state dict."""
    q_model = build_quantized_shell()
    q_model.load_state_dict(q_state_dict, strict=False)

    f_model = SentenceCNN()
    qmods = dict(q_model.named_modules())
    fmods = dict(f_model.named_modules())
    for name in ["conv1", "conv2", "conv3", "fc1", "fc2"]:
        qmod = qmods[name]
        fmod = fmods[name]
        qweight = qmod.weight()
        fmod.weight.data = qweight.dequantize() if hasattr(qweight, "is_quantized") and qweight.is_quantized else qweight.float()
        if hasattr(qmod, "bias") and qmod.bias() is not None:
            fmod.bias.data = qmod.bias().float()
    f_model.eval()
    return f_model


def load_model(weight_path: str, device: torch.device):
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    if "int8" in weight_path:
        model = dequantize_to_float_model(state)
        mode = "int8_dequant_float"
    else:
        model = SentenceCNN()
        model.load_state_dict(state, strict=False)
        mode = "fp32"

    model.to(device).eval()
    return model, mode


def main():
    parser = argparse.ArgumentParser(description="Dump single-sample logits for C-side verification")
    parser.add_argument("--csv", type=str, default="embeddings/test_rest_int8.csv", help="Embedding CSV (q0~q383, label, text)")
    parser.add_argument("--row", type=int, default=2658, help="Row index (0-based) to use")
    parser.add_argument("--weights", type=str, default="weights/sentence_cnn_int8.pth", help="Model weights (fp32 or quantized checkpoint)")
    parser.add_argument("--out_txt", type=str, default="reference_logit.txt", help="Where to write the reference logit")
    args = parser.parse_args()

    device = torch.device("cpu")
    df = pd.read_csv(args.csv)
    features = [f"q{i}" for i in range(384)]

    idx = max(0, min(args.row, len(df) - 1))
    row = df.iloc[idx]
    x_np = row[features].to_numpy(dtype="float32")
    label = int(row["label"])
    text = row.get("text", "")

    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

    model, mode = load_model(args.weights, device)
    with torch.no_grad():
        # pre-sigmoid logit
        logits = model.fc2(F.relu(model.fc1(
            model.pool(
                F.relu(model.conv3(
                    F.relu(model.conv2(
                        F.relu(model.conv1(x.unsqueeze(1)))
                    ))
                ))
            ).squeeze(-1)
        ))).squeeze().item()
        prob = 1.0 / (1.0 + math.exp(-logits))
        logit_val = logits
        print(f"[debug] logit={logit_val:.9f} prob={prob:.9f}")

    with open(args.out_txt, "w") as f:
        f.write(f"source_csv: {args.csv}\n")
        f.write(f"row_index: {idx}\n")
        f.write(f"text: {text}\n")
        f.write(f"label: {label}\n")
        f.write(f"weights: {args.weights} ({mode})\n")
        f.write(f"logit: {logit_val:.6f}\n")
        f.write(f"prob: {prob:.6f}\n")

    print("✅ Reference logit written to", args.out_txt)
    print(f"row={idx}, mode={mode}, logit={logit_val:.6f}, prob={prob:.6f}")


if __name__ == "__main__":
    main()
