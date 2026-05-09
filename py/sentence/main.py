import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.quantization as tq
import os
import argparse

from sentence_infer import SentenceCNN, SentenceBERTDataset, load_data, create_data_loaders, train_model, evaluate_model, plot_training_history

import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import onnxruntime as ort


def export_quantized_weights(model_int8, filename="weights_q.h"):
    with open(filename, "w") as f:
        f.write("#ifndef WEIGHTS_Q_H\n#define WEIGHTS_Q_H\n\n#include <stdint.h>\n\n")

        prev_scale = 1.0  # 輸入已是 int8，對 C 端而言假設 zp=0
        layer_idx = 1

        ordered_layers = ["conv1", "conv2", "conv3", "fc1", "fc2"]
        modules = dict(model_int8.named_modules())

        for name in ordered_layers:
            module = modules.get(name)
            if module is None or not hasattr(module, "weight") or module.weight() is None:
                raise RuntimeError(f"Missing quantized module: {name}")

            w_q = module.weight()
            w_int8 = w_q.int_repr().cpu().numpy()
            qscheme = w_q.qscheme()

            # === 取 weight scale ===
            if qscheme in (torch.per_tensor_symmetric, torch.per_tensor_affine):
                scales = np.array([float(w_q.q_scale())], dtype=np.float32)
            elif qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                scales = w_q.q_per_channel_scales().cpu().numpy().astype(np.float32)
            else:
                raise RuntimeError(f"Unsupported qscheme: {qscheme}")

            out_ch = len(scales)
            s_in = prev_scale
            s_out = float(getattr(module, "scale", 1.0))
            z_out = int(getattr(module, "zero_point", 0))
            M = (s_in * scales) / s_out

            flat = w_int8.flatten()
            f.write(f"// Layer {layer_idx}: {name}, shape={w_int8.shape}, qscheme={qscheme}\n")
            f.write(f"// scale_in={s_in:.8f}, scale_out={s_out:.8f}, zp_out={z_out}\n\n")

            # === 匯出 scale ===
            f.write(f"const float {name.replace('.', '_')}_scales[{out_ch}] = "
                    "{" + ",".join(f"{v:.8f}" for v in scales) + "};\n")

            # === 匯出 output zero_point (C 端 requantize 使用) ===
            zps = np.full(out_ch, z_out, dtype=np.int32)
            f.write(f"const int32_t {name.replace('.', '_')}_zero_points[{out_ch}] = "
                    "{" + ",".join(map(str, zps)) + "};\n")

            # === 匯出每通道 M ===
            f.write(f"const float {name.replace('.', '_')}_M[{out_ch}] = "
                    "{" + ",".join(f"{v:.8f}" for v in M) + "};\n")

            # === 匯出權重 ===
            f.write(f"const int8_t {name.replace('.', '_')}_weight[{flat.size}] = "
                    "{" + ",".join(map(str, flat)) + "};\n\n")

            # === 匯出 bias ===
            if hasattr(module, "bias") and module.bias() is not None:
                b_fp32 = module.bias().detach().cpu().numpy()
                if len(scales) > 1:
                    b_int32 = np.round(b_fp32 / (s_in * scales)).astype(np.int32)
                else:
                    b_int32 = np.round(b_fp32 / (s_in * scales[0])).astype(np.int32)

                flat_b = b_int32.flatten()
                f.write(f"// {name}.bias (int32), 原始 fp32={b_fp32.tolist()}\n")
                f.write(f"const int32_t {name.replace('.', '_')}_bias[{flat_b.size}] = "
                        "{" + ",".join(map(str, flat_b)) + "};\n\n")

            prev_scale = s_out  # 下一層輸入 scale
            layer_idx += 1

        f.write("#endif // WEIGHTS_Q_H\n")
    print(f"✅ Quantized weights (per-channel + zero_point) exported to {filename}")

def export_quantized_weights_gemmini(model_int8, filename="weights_q_gemmini.h"):
    with open(filename, "w") as f:
        f.write("#ifndef WEIGHTS_Q_GEMMINI_H\n#define WEIGHTS_Q_GEMMINI_H\n\n#include <stdint.h>\n\n")

        prev_scale = 1.0  # 輸入已是 int8，對 Gemmini/RVV 假設 zp=0
        layer_idx = 1

        ordered_layers = ["conv1", "conv2", "conv3", "fc1", "fc2"]
        modules = dict(model_int8.named_modules())

        for name in ordered_layers:
            module = modules.get(name)
            if module is None or not hasattr(module, "weight") or module.weight() is None:
                raise RuntimeError(f"Missing quantized module: {name}")

            w_q = module.weight()
            w_int8 = w_q.int_repr().cpu().numpy()
            qscheme = w_q.qscheme()

            # === 取 weight scale ===
            if qscheme in (torch.per_tensor_symmetric, torch.per_tensor_affine):
                scales = np.array([float(w_q.q_scale())], dtype=np.float32)
            elif qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
                scales = w_q.q_per_channel_scales().cpu().numpy().astype(np.float32)
            else:
                raise RuntimeError(f"Unsupported qscheme: {qscheme}")

            out_ch = len(scales)
            s_in = prev_scale
            s_out = float(getattr(module, "scale", 1.0))
            z_out = int(getattr(module, "zero_point", 0))
            M = (s_in * scales) / s_out

            f.write(f"// Layer {layer_idx}: {name}, shape={w_int8.shape}, qscheme={qscheme}\n")
            f.write("// Gemmini input layout: NHWC\n")
            f.write("// Gemmini weight layout: O H W I\n")
            f.write(f"// scale_in={s_in:.8f}, scale_out={s_out:.8f}, zp_out={z_out}\n\n")

            # === 匯出 scale ===
            f.write(f"const float {name.replace('.', '_')}_scales[{out_ch}] = "
                    "{" + ",".join(f"{v:.8f}" for v in scales) + "};\n")

            # === 匯出 output activation zero_point ===
            zps_out = np.full(out_ch, z_out, dtype=np.int32)
            f.write(f"elem_t {name.replace('.', '_')}_zero_points[{out_ch}] = "
                    "{" + ",".join(map(str, zps_out)) + "};\n")

            # === 匯出每通道 M ===
            f.write(f"const float {name.replace('.', '_')}_M[{out_ch}] = "
                    "{" + ",".join(f"{v:.8f}" for v in M) + "};\n")

            # === 匯出權重 (Gemmini OHWI) ===
            weight_name = f"{name.replace('.', '_')}_weight"
            if w_int8.ndim == 4:
                # O I H W -> O H W I
                w_ohwi = np.transpose(w_int8, (0, 2, 3, 1))
            elif w_int8.ndim == 3:
                # O I K -> O 1 K I
                w_oki = np.transpose(w_int8, (0, 2, 1))
                w_ohwi = w_oki[:, np.newaxis, :, :]
            elif w_int8.ndim == 2:
                # FC: O I -> O 1 1 I
                w_ohwi = w_int8[:, np.newaxis, np.newaxis, :]
            else:
                raise RuntimeError(f"Unsupported weight ndim: {w_int8.ndim}")

            o_dim, h_dim, w_dim, i_dim = w_ohwi.shape
            f.write(f"elem_t {weight_name}[{o_dim}][{h_dim}][{w_dim}][{i_dim}] = {{\n")
            for o in range(o_dim):
                f.write("    {\n")
                for h in range(h_dim):
                    f.write("        {\n")
                    for w in range(w_dim):
                        f.write("            {" + ",".join(map(str, w_ohwi[o, h, w, :])) + "},\n")
                    f.write("        },\n")
                f.write("    },\n")
            f.write("};\n\n")

            # === 匯出 bias ===
            if hasattr(module, "bias") and module.bias() is not None:
                b_fp32 = module.bias().detach().cpu().numpy()
                if len(scales) > 1:
                    b_int32 = np.round(b_fp32 / (s_in * scales)).astype(np.int32)
                else:
                    b_int32 = np.round(b_fp32 / (s_in * scales[0])).astype(np.int32)

                flat_b = b_int32.flatten()
                f.write(f"// {name}.bias (int32), 原始 fp32={b_fp32.tolist()}\n")
                f.write(f"acc_t {name.replace('.', '_')}_bias[{flat_b.size}] = "
                        "{" + ",".join(map(str, flat_b)) + "};\n\n")

            prev_scale = s_out  # 下一層輸入 scale
            layer_idx += 1

        f.write("#endif // WEIGHTS_Q_GEMMINI_H\n")
    print(f"✅ Quantized weights (Gemmini layout) exported to {filename}")


def _write_nested_c_array(f, c_type, name, arr):
    arr = np.asarray(arr)
    dims = "".join(f"[{dim}]" for dim in arr.shape)
    f.write(f"{c_type} {name}{dims} = ")

    def emit(x, indent):
        if x.ndim == 1:
            f.write("{" + ",".join(str(int(v)) for v in x) + "}")
            return
        f.write("{\n")
        for item in x:
            f.write(" " * indent)
            emit(item, indent + 4)
            f.write(",\n")
        f.write(" " * (indent - 4) + "}")

    emit(arr, 4)
    f.write(";\n\n")


def _conv1d_to_gemmini_hwio_square(w_int8):
    if w_int8.ndim != 3:
        raise RuntimeError(f"Expected Conv1d weight O,I,K, got {w_int8.shape}")
    out_ch, in_ch, kernel = w_int8.shape
    hwio = np.zeros((kernel, kernel, in_ch, out_ch), dtype=np.int8)
    center_h = kernel // 2
    for out_idx in range(out_ch):
        for in_idx in range(in_ch):
            hwio[center_h, :, in_idx, out_idx] = w_int8[out_idx, in_idx, :]
    return hwio


def export_gemmini_native_weights(model_int8, filename="weights_gemmini_native.h"):
    modules = dict(model_int8.named_modules())
    ordered_layers = ["conv1", "conv2", "conv3", "fc1", "fc2"]
    prev_scale = 1.0
    requant_scales = {}

    for name in ordered_layers:
        module = modules.get(name)
        if module is None or not hasattr(module, "weight") or module.weight() is None:
            raise RuntimeError(f"Missing quantized module: {name}")

        w_q = module.weight()
        qscheme = w_q.qscheme()
        if qscheme not in (torch.per_tensor_symmetric, torch.per_tensor_affine):
            raise RuntimeError(
                f"{name} is {qscheme}; rerun calibration with --calibration_qscheme per_tensor_gemmini"
            )

        s_weight = float(w_q.q_scale())
        s_out = float(getattr(module, "scale", 1.0))
        requant_scales[name] = (prev_scale * s_weight) / s_out
        prev_scale = s_out

    conv1_hwio = _conv1d_to_gemmini_hwio_square(modules["conv1"].weight().int_repr().cpu().numpy())
    conv2_hwio = _conv1d_to_gemmini_hwio_square(modules["conv2"].weight().int_repr().cpu().numpy())
    conv3_hwio = _conv1d_to_gemmini_hwio_square(modules["conv3"].weight().int_repr().cpu().numpy())
    pool_identity_1x1_hwio = np.zeros((1, 1, 256, 256), dtype=np.int8)
    for idx in range(256):
        pool_identity_1x1_hwio[0, 0, idx, idx] = 1

    guard = os.path.basename(filename).upper().replace(".", "_").replace("-", "_")
    with open(filename, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("// Generated for Gemmini-native timing path.\n")
        f.write("// Requires per-tensor symmetric weight calibration.\n")
        f.write("// Conv1d weights are embedded as square HWIO kernels for tiled_conv_auto.\n\n")

        for name in ordered_layers:
            macro = f"{name.upper()}_REQUANT_SCALE"
            f.write(f"#ifndef {macro}\n#define {macro} {requant_scales[name]:.8f}f\n#endif\n")
        f.write("\n")

        _write_nested_c_array(f, "elem_t", "conv1_weight_hwio", conv1_hwio)
        _write_nested_c_array(f, "elem_t", "conv2_weight_hwio", conv2_hwio)
        _write_nested_c_array(f, "elem_t", "conv3_weight_hwio", conv3_hwio)
        _write_nested_c_array(f, "elem_t", "pool_identity_1x1_hwio", pool_identity_1x1_hwio)
        f.write(f"#endif // {guard}\n")

    print(f"✅ Gemmini native per-tensor weights exported to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Train sentence validity classification model')
    parser.add_argument('--csv_file', type=str, default="embeddings/test_rest_int8.csv", help='Training data CSV file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_model', type=str, default='sentence_cnn_model.pth', help='Model save filename')
    parser.add_argument('--save_weights', type=str, default='weights/best_weights.pth', help='Save best weights filename')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'calibration', 'test', 'onnx'], default='train', help='Run mode: train or inference')
    parser.add_argument('--calibration_qscheme', type=str, choices=['per_channel', 'per_tensor_gemmini'], default='per_tensor_gemmini', help='Weight observer for calibration export')

    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    print("-" * 60)
    print("Starting data loading and preprocessing...")
    
    # Load data
    try:
        X_train, X_test, y_train, y_test = load_data(args.csv_file, test_size=args.test_size)
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        
        # Check label distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Training set label distribution: {dict(zip(unique, counts))}")
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"Test set label distribution: {dict(zip(unique, counts))}")
        
    except FileNotFoundError:
        print(f"Error: File not found {args.csv_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, batch_size=args.batch_size
    )
    
    print("-" * 60)
    print("Initializing model...")
    
    # Create model
    model = SentenceCNN(input_dim=384, num_classes=1, dropout_rate=args.dropout)
    
    if args.mode == "train":
        # Print model structure
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("-" * 60)
        print("Starting training...")
        
        # Train model with automatic weight saving
        model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
            model, train_loader, test_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.lr, 
            device=device,
            save_weights=True,
            weight_save_path=args.save_weights
        )
        
        print("-" * 60)
        print("Training complete, evaluating model...")
        
        # Evaluate model
        final_accuracy = evaluate_model(model, test_loader, device=device)
        
        print("-" * 60)
        print("Plotting training history...")
        
        # Plot training history
        plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)
        
        print("-" * 60)
        print("Saving model...")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 384,
                'num_classes': 1,
                'dropout_rate': args.dropout
            },
            'training_config': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'test_size': args.test_size
            },
            'final_accuracy': final_accuracy,
            'training_history': {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }
        }, args.save_model)
        
        print(f"Model saved to: {args.save_model}")
        print(f"Final test accuracy: {final_accuracy:.4f}")
        
        print("-" * 60)
        print("Training complete!")

    elif args.mode == "inference":
        print("🔍 Inference mode")
        checkpoint = torch.load(args.save_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()

        # === 讀整份 CSV ===
        try:
            df = pd.read_csv(args.csv_file)
            X = df[[f"q{i}" for i in range(384)]].values.astype("float32")
            y = df["label"].values.astype("int64") if "label" in df.columns else None
        except Exception as e:
            print(f"❌ Error reading inference CSV: {e}")
            return

        # === 用跟訓練時完全一樣的切分方式 ===
        if y is not None:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_test = X
            y_test = np.zeros(len(X_test), dtype=np.int64)

        print(f"🧾 Test set size: {len(X_test)} samples")

        # === 建立 DataLoader ===
        dataset = SentenceBERTDataset(
            None, test_mode=True, X_data=X_test, y_data=y_test
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # === Inference ===
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x).squeeze()
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)

        df_test = pd.DataFrame(X_test, columns=[f"q{i}" for i in range(384)])
        df_test["label"] = y_test
        df_test["prediction"] = np.array(all_preds, dtype=int)

        # === 若有標籤，輸出評估 ===
        if y is not None:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            import seaborn as sns, matplotlib.pyplot as plt

            acc = accuracy_score(y_test, df_test["prediction"])
            print(f"📈 Inference Accuracy: {acc:.4f}")
            print("\n📋 Classification Report:")
            print(classification_report(y_test, df_test["prediction"], target_names=['Invalid', 'Valid']))

            cm = confusion_matrix(y_test, df_test["prediction"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Invalid', 'Valid'], yticklabels=['Invalid', 'Valid'])
            plt.title("Confusion Matrix (Inference)")
            plt.savefig("inference_confusion_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("💾 Confusion matrix saved as inference_confusion_matrix.png")

        out_file = "inference_results.csv"
        df_test.to_csv(out_file, index=False)
        print(f"✅ Inference done! Results saved to {out_file}")

        
    elif args.mode == "calibration":
        print("Calibration mode for PTQ use cpu only")
        device = torch.device('cpu')
        # 載入模型
        checkpoint = torch.load(args.save_weights, map_location="cpu", weights_only=True)
        model_fp32 = SentenceCNN()
        model_fp32.load_state_dict(checkpoint["model_state_dict"])
        model_fp32.eval()

        if args.calibration_qscheme == "per_tensor_gemmini":
            weight_observer = tq.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
            int8_checkpoint = "weights/sentence_cnn_int8_gemmini_per_tensor.pth"
            c_header = "weights_q_gemmini_per_tensor_ref.h"
            gemmini_header = "weights_q_gemmini_per_tensor.h"
            native_header = "weights_gemmini_native_per_tensor.h"
            print("Calibration qscheme: per-tensor symmetric weights for Gemmini")
        else:
            weight_observer = tq.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
            int8_checkpoint = "weights/sentence_cnn_int8.pth"
            c_header = "weights_q.h"
            gemmini_header = "weights_q_gemmini.h"
            native_header = None
            print("Calibration qscheme: per-channel symmetric weights")

        # fbgemm 適用 x86 看用哪種統計模型來做觀測
        model_fp32.qconfig = tq.QConfig(
            activation=tq.HistogramObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric  # C 端只支援 zp=0
            ),
            weight=weight_observer
        )
        # 自動插入 observer
        tq.prepare(model_fp32, inplace=True)

        # read csv default header
        df = pd.read_csv("embeddings/test_rest_int8.csv")
        X = df[[f"q{i}" for i in range(384)]].values.astype("float32")
        y = df["label"].values.astype("int64") if "label" in df.columns else None
        # 建立 Dataset + DataLoader
        calib_dataset = SentenceBERTDataset(
            None, test_mode=True,
            X_data=X, y_data=y if y is not None else np.zeros(len(X))
        )
        calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=True)

        # 只取 2000 筆有代表性的資料做校正
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(calib_loader):
                model_fp32(batch_x.to(device))
                if i * 32 >= 2000:
                    break

        # 轉換成 INT8 模型
        """
        取observer的
        """
        model_int8 = tq.convert(model_fp32, inplace=False)

        export_quantized_weights(model_int8, c_header)
        print(f"✅ Quantized weights exported to {c_header}")
        torch.save({
        "model_state_dict": model_int8.state_dict(),
        "qconfig": str(model_fp32.qconfig),
        }, int8_checkpoint)
        
        
        export_quantized_weights_gemmini(model_int8, gemmini_header)
        print(f"✅ Quantized weights exported to {gemmini_header}")
        if native_header is not None:
            export_gemmini_native_weights(model_int8, native_header)
        for name, module in model_int8.named_modules():
            if hasattr(module, "scale") and hasattr(module, "zero_point"):
                print(f"{name}: scale={module.scale}, zp={module.zero_point}")

        print("✅ Calibration finished, quantized model ready!")

    elif args.mode == "test":
        print("Test mode - not implemented yet")
        model = SentenceCNN()
        dummy_input = torch.randn(1, 384)
        output = model(dummy_input)
    elif args.mode == "onnx":
        print("ONNX export mode")
        checkpoint = torch.load(args.save_weights, map_location=device)
        model = SentenceCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()

        dummy_input = torch.randn(1, 384).to(device)
        onnx_file = "sentence_cnn.onnx"
        torch.onnx.export(model, dummy_input, onnx_file, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                          opset_version=11)
        print(f"✅ Model exported to ONNX format: {onnx_file}")

    
if __name__ == "__main__":
    main()
