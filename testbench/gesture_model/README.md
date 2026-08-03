## Build and run

Linux/Spike and bare-metal compile the same
`testbench/gesture_model/gesture_recognition_fp32.c` source and execute the
same `main()`. Bare-metal replaces only startup, linking, minilib, and the
`nn_runtime` implementation; there is no separate application adapter.
The bare-metal startup prints hardware and RVV/VLEN information before entering
this shared `main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu gesture_recognition_fp32 default
./scripts/configure_build.sh linux-pk vector gesture_recognition_fp32 zvl128b
```

Run the CPU or RVV executable with Spike:

```sh
spike --isa=rv64gc_zicntr_zihpm \
    pk build/linux-pk/gesture_recognition_fp32/default/cpu/static/gesture_recognition_fp32

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/gesture_recognition_fp32/zvl128b/vector/static/gesture_recognition_fp32
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector gesture_model V128D128B
```

Flash it to an SD card:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=gesture_model \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. The flash target
rejects partitions and mounted devices, then writes the binary beginning at
512-byte block 34. This operation overwrites data on the selected device.

## Spike Performance

* spike `default CPU clock = 1GHz`

* ISA `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d`

| Metric | VLEN=128 | VLEN=256 | VLEN=512 | Notes |
| --- | --- | --- | --- | --- |
| Cycles | 1720149 | 902544 | 508672 | |
| CPI |  |  |
| Runtime |  |  |
