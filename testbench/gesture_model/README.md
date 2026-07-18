## Build commands

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu gesture_recognition_fp32 default
./scripts/configure_build.sh linux-pk vector gesture_recognition_fp32 zvl128b
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector gesture_model V128D128B
```

## Spike Performance

* spike `default CPU clock = 1GHz`

* ISA `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d`

| Metric | VLEN=128 | VLEN=256 | VLEN=512 | Notes |
| --- | --- | --- | --- | --- |
| Cycles | 1720149 | 902544 | 508672 | |
| CPI |  |  |
| Runtime |  |  |
