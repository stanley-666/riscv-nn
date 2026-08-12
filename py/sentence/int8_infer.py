# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np
import pandas as pd
import torch
import torch.quantization as tq
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sentence_infer import SentenceCNN, SentenceBERTDataset


def build_quantized_model(device: torch.device, qscheme: str) -> torch.nn.Module:
    """
    Recreate the static-quantized SentenceCNN structure used during calibration,
    then load quantized weights into it.
    """
    model = SentenceCNN()
    if qscheme == "per_tensor_gemmini":
        weight_observer = tq.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )
    elif qscheme == "per_channel":
        weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )
    else:
        raise ValueError(f"Unsupported qscheme: {qscheme}")

    model.qconfig = tq.QConfig(
        activation=tq.HistogramObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        ),
        weight=weight_observer,
    )
    # Insert observers and immediately convert to get matching quantized module types,
    # so the checkpoint keys line up.
    tq.prepare(model, inplace=True)
    tq.convert(model, inplace=True)
    model.to(device).eval()
    return model

def dequantize_to_float_model(model_int8: torch.nn.Module) -> torch.nn.Module:
    """
    Build a fresh float SentenceCNN and copy dequantized weights/biases from the
    quantized model. This bypasses missing quantized kernels (e.g., adaptive_max_pool1d).
    """
    float_model = SentenceCNN()
    qmods = dict(model_int8.named_modules())
    fmods = dict(float_model.named_modules())

    for name in ["conv1", "conv2", "conv3", "fc1", "fc2"]:
        qmod = qmods[name]
        fmod = fmods[name]

        # Weight
        qweight = qmod.weight()
        fmod.weight.data = qweight.dequantize() if qweight.is_quantized else qweight.float()

        # Bias
        if hasattr(qmod, "bias") and qmod.bias() is not None:
            fmod.bias.data = qmod.bias().float()

    float_model.eval()
    return float_model


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[[f"q{i}" for i in range(384)]].values.astype("float32")
    y = df["label"].values.astype("int64") if "label" in df.columns else None

    dataset = SentenceBERTDataset(
        None, test_mode=True, X_data=X, y_data=y if y is not None else np.zeros(len(X))
    )
    return dataset, X, y


def run_inference(csv_path: str, weight_path: str, batch_size: int, out_csv: str, qscheme: str):
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    # Build quantized model shell then load quantized weights produced by calibration.
    model = build_quantized_model(device, qscheme)
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)

    dataset, X, y = load_dataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Fallback: dequantize weights into a fresh float model to avoid missing quantized kernels.
    model = dequantize_to_float_model(model)
    model.to(device)
    print(f"ℹ️ Quantized checkpoint loaded ({qscheme}); running inference in float fallback (dequantized weights).")

    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()
            batch_preds = (outputs > 0.5).float().cpu().numpy()
            preds.extend(batch_preds)

    preds = np.array(preds, dtype=int)
    df_out = pd.DataFrame(X, columns=[f"q{i}" for i in range(384)])
    if y is not None:
        df_out["label"] = y
    df_out["prediction"] = preds
    df_out.to_csv(out_csv, index=False)

    print(f"✅ INT8 inference done. Results saved to {out_csv}")

    # Optional eval if labels are present
    if y is not None:
        acc = accuracy_score(y, preds)
        print(f"📈 Accuracy: {acc:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y, preds, target_names=["Invalid", "Valid"]))

        cm = confusion_matrix(y, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Invalid", "Valid"],
            yticklabels=["Invalid", "Valid"],
        )
        plt.title("Confusion Matrix (INT8 Inference)")
        plt.savefig("inference_confusion_matrix_int8.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("💾 Confusion matrix saved as inference_confusion_matrix_int8.png")


def main():
    parser = argparse.ArgumentParser(description="INT8 inference for quantized SentenceCNN")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="embeddings/test_rest_int8.csv",
        help="CSV containing q0~q383 and optional label column",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="weights/sentence_cnn_int8_gemmini_per_tensor.pth",
        help="Quantized checkpoint produced by calibration",
    )
    parser.add_argument(
        "--qscheme",
        type=str,
        choices=["per_tensor_gemmini", "per_channel"],
        default="per_tensor_gemmini",
        help="Quantized shell layout used to load the checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="inference_results_int8.csv",
        help="Where to save predictions",
    )
    args = parser.parse_args()

    run_inference(args.csv_file, args.weight_path, args.batch_size, args.out_csv, args.qscheme)


if __name__ == "__main__":
    main()
