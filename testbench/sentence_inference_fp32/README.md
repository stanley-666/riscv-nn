## Spike Performance
* spike `default CPU clock = 1GHz`

* ISA `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d`

| Metric | VLEN=128 | VLEN=256 | VLEN=512 | Notes |
| --- | --- | --- | --- | --- |
| Cycles | 10083273 | 5099263 | 2629576 | |
| CPI |  |  |
| Runtime |  |  |

## Keeping C vs Python logits aligned
Before比較/驗證，請同步權重與 embedding：

1. 匯出最新 FP32 權重：
```
python py/sentence/export_fp32_header.py \
  --ckpt py/sentence/weights/best_weights.pth \
  --out testbench/sentence_inference_fp32/weights_fp32.h
```

2. 產生目標 row 的 embedding（例：2658）並覆蓋本目錄：
```
python py/sentence/dump_embeddings.py --row 2658 --csv py/sentence/embeddings/test_rest_int8.csv --prefix random_embedding
cp py/sentence/random_embedding.h testbench/sentence_inference_fp32/random_embedding.h
```

3. 重編並執行：
```
make clean && make sentence_inference_fp32
./sentence_inference_fp32
```

Python 比對（同 row/權重）：
```
python py/sentence/dump_logit.py --csv py/sentence/embeddings/test_rest_int8.csv --row 2658 --weights py/sentence/weights/best_weights.pth
```
