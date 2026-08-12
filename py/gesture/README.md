<!-- SPDX-FileContributor: Person: Stanley Lee -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# gesture

這個專案包含一個 1D CNN 的手勢分類模型與一系列工具，涵蓋：
TensorFlow / PyTorch 推論、BatchNorm 融合、以及匯出給 C 端使用的權重檔。

## 重點內容
- TensorFlow 模型架構與自訂 Focal Loss：`architecture.py`
- 內建手勢樣本推論（TF）：`tf_run_sample.py`
- PyTorch 推論與 Keras 權重載入：`torch_fused_model.py`
- Conv+BN 融合（產生不含 BN 的模型）：`fuse_bn.py`
- 匯出 C header 權重（float32）：`export_fused_h5_to_header.py`
- TF / Torch 輸出比對：`compare_inference.py`

## 環境與安裝
需求：Python 3.10、TensorFlow、PyTorch、NumPy。

選一種方式即可：

1) 使用專案提供的 `build.sh` 建立本地 Conda 環境（路徑在 `.conda-envs/tf`）：
```bash
./build.sh
conda activate ./.conda-envs/tf
```

2) 使用全域 Conda 環境名稱 `tf`（對應 `env.sh` 的設定）：
```bash
source env.sh
```

## 快速推論
### TensorFlow（使用內建手勢樣本）
```bash
python tf_run_sample.py --model model.h5 --layout wc --show-summary
```

### PyTorch（載入融合後的 Keras 權重）
```bash
python torch_fused_model.py --weights fused_model.h5
```

## BN 融合與 C 權重匯出
### 1) 融合 Conv + BatchNorm
```bash
python fuse_bn.py --model-path model.h5 --output-path fused_model.h5
```

### 2) 匯出 C header 權重
```bash
python export_fused_h5_to_header.py --model fused_model.h5 --output weights_fused_fp32.h
```

## TF / Torch 推論一致性比對
```bash
python compare_inference.py \
  --tf-model fused_model.h5 \
  --torch-weights fused_model.h5
```

## 輸入格式說明
- 主要模型預期輸入為 5 個通道、窗口長度 50 的序列。
- TensorFlow：`[N, W, C]`（時間在前，通道在後）。
- PyTorch：`[N, C, W]`（通道在前，時間在後）。
- 範例程式會將原始值除以 `360.0` 做縮放。

## 專案檔案簡表
- `model.h5`：原始 Keras 模型
- `fused_model.h5`：已融合 BN 的 Keras 模型
- `weights_fused_fp32.h`：匯出給 C 端使用的權重
