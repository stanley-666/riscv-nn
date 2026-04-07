`sentence_inference_fp32_time`

FP32 sentence inference testbench variant that measures latency with `time.h`
only. This version intentionally does not use `rdcycle`.

This directory is self-contained and includes:

- `sentence_inference_fp32_time.c`
- `weights_fp32.h`
- `random_embedding.h`
- `test_dataset_2658.h`
- `test_dataset.h`

Build example:

```bash
make CONFIG=zvl512b sentence_inference_fp32_time
```
