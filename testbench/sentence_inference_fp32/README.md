## Build commands

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu sentence_inference_fp32 default
./scripts/configure_build.sh linux-pk vector sentence_inference_fp32 zvl128b
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector sentence_inference_fp32 V128D128B
```

## Spike Performance
* spike `default CPU clock = 1GHz`

* ISA `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d`

| Metric | VLEN=128 | VLEN=256 | VLEN=512 | Notes |
| --- | --- | --- | --- | --- |
| Cycles | 10083273 | 5099263 | 2629576 | |
| CPI |  |  |
| Runtime |  |  |


## logits before sigmoid

C test data 2658th :  11.3552
Python test data 2658th : 11.355240

Inference time: 7.949469 seconds
Overall accuracy: 99.25% (2638/2658 correct)
