## Download training dataset if you want to re-train the sentence model
[Please Download embeddings dataset here](https://drive.google.com/drive/folders/1hmy77IVeroNg7Xs1Iwtk22LRgiWlB4Sn?usp=drive_link)


## QUICKSTART

### Train

```python
python main.py --mode train
```

### Inference

```python
python main.py --mode inference
```

### Calibration
```python
python main.py --mode calibration
```

### Gemmini per-tensor calibration
```python
python main.py --mode calibration --calibration_qscheme per_tensor_gemmini
```

This reruns calibration with per-tensor symmetric weights for Gemmini. It emits:
`weights/sentence_cnn_int8_gemmini_per_tensor.pth`,
`weights_q_gemmini_per_tensor.h`, and `weights_gemmini_native_per_tensor.h`.

Use `weights_gemmini_native_per_tensor.h` for the Gemmini native timing path,
then keep `weights_q_gemmini_per_tensor.h` as the matching bias/FC/reference
metadata. The original per-channel headers remain separate.

### INT8 Inference
```python
# 使用校正後的同一份權重；PyTorch 沒有 AdaptiveMaxPool1d 的量化 kernel，
# 腳本會載入 int8 權重後解量化為 FP32 推論（權重仍來自 calibration）
python int8_infer.py --csv_file embeddings/test_rest_int8.csv --weight_path weights/sentence_cnn_int8.pth --batch_size 32
```
> 若要純 INT8 路徑，請將模型的 `AdaptiveMaxPool1d` 改成有量化 kernel 的 `MaxPool1d`（kernel/stride 使輸出長度=1），重新校正/量化後再推論。

### Single-sample logit for C-side check（pre-sigmoid）
```python
python dump_logit.py --csv embeddings/test_rest_int8.csv --row 2658 --weights weights/sentence_cnn_int8.pth
# 輸出 reference_logit.txt，含 pre-sigmoid logit 與 prob
```

### Gemmini timing-oriented export from an existing checkpoint
```python
python export_gemmini_timing_header.py \
  --weights weights/sentence_cnn_int8.pth \
  --out weights_gemmini_timing.h \
  --scale-mode first
```

This export is for Gemmini dataflow latency experiments, not accuracy parity with
the RVV/per-channel path. It emits HWIO Conv1d weights embedded into square 2D
kernels, scalar per-tensor requant scales for Gemmini store scaling, and a 1x1
identity kernel for staged Gemmini pooling.

## Quantization strategy
- Post-training static quantization with PyTorch observers (`torch.quantization`): `HistogramObserver` for activations (qint8, per-tensor symmetric, zero-point forced to 0) and `PerChannelMinMaxObserver` for weights (qint8, per-channel symmetric).
- Calibration runs on CPU over `embeddings/test_rest_int8.csv` (first ~2000 samples) to insert observers and convert the trained FP32 `SentenceCNN` into INT8.
- Running `python main.py --mode calibration` exports C-friendly weights/metadata (`weights_q.h` for generic C, `weights_q_gemmini.h` for Gemmini/RVV layout) and an INT8 checkpoint at `weights/sentence_cnn_int8.pth`.

## Requirements
- Python 3.x, PyTorch with CPU quantization backend enabled (FBGEMM for x86).
- Python packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `onnx`, `onnxruntime`, `tqdm` (optional progress bars).
- Training/calibration CSVs expected under `py/sentence/embeddings/` (`train_int8.csv`, `test_rest_int8.csv`) generated from llama.cpp.
