## Build commands

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu sentence_inference_int8 default
./scripts/configure_build.sh linux-pk vector sentence_inference_int8 zvl128b
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector sentence_inference_int8 V128D128B
```

## Spike Performance
* spike `default CPU clock = 1GHz`

* ISA `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d`

| Metric | VLEN=128 | VLEN=256 | VLEN=512 | Notes |
| --- | --- | --- | --- | --- |
| Cycles | 10083273 | 5099263 | 2629576 | |
| CPI |  |  |
| Runtime |  |  |
