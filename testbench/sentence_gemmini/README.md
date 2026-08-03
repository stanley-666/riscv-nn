# Gemmini SentenceCNN testbench

Copyright 2026 Stanley Lee
SPDX-License-Identifier: Apache-2.0

This testbench runs the INT8 SentenceCNN through the default 16x16 Gemmini
configuration. The same inference source is compiled in two environments:

- Linux ABI with Spike `pk` and `--extension=gemmini`;
- freestanding bare-metal with the repository startup, linker, minilib,
  syscalls, and trap runtime.

Both targets compile `testbench/sentence_gemmini/sentence_gemmini.c` and
execute the same `main()`. Bare-metal replaces the platform runtime and link
environment; there is no separate application adapter.
The bare-metal startup prints hardware information before entering this shared
`main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

The executable entry is `sentence_gemmini.c`. Gemmini ISA and tiling headers
remain under `csrc/backends/riscv/gemmini/ops/`, while the generated input and
weight headers are under its `bareMetalC/` data directory.

The default configuration uses:

- per-tensor INT8 model headers;
- native `tiled_conv_auto` with square HWIO kernels;
- Gemmini store-scale requantization and fused ReLU;
- staged Gemmini pooling;
- Gemmini fully connected layers;
- host processing only for final sigmoid and logging.

Build and execute the Linux/Spike target:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh
./scripts/configure_build.sh linux-pk gemmini sentence_gemmini default
/usr/bin/cmake --build \
    build/cmake/linux-pk-sentence_gemmini-gemmini-default \
    --target run-spike-gemmini
```

Equivalent direct execution:

```sh
spike --isa=rv64gc_zicntr_zihpm --extension=gemmini \
    "$RISCV/riscv64-unknown-elf/bin/pk" \
    build/linux-pk/sentence_gemmini/default/gemmini/static/sentence_gemmini
```

Build the flashable bare-metal image:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh
./scripts/configure_build.sh baremetal gemmini sentence_gemmini GEMMINI

/usr/bin/cmake --build \
    build/cmake/baremetal-sentence_gemmini-gemmini-GEMMINI \
    --target baremetal-dump
```

Flash it to an SD card:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=sentence_gemmini \
    BACKEND=gemmini \
    HARDWARE_CONFIG=GEMMINI \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. The flash target
rejects partitions and mounted devices, then writes the binary beginning at
512-byte block 34. This operation overwrites data on the selected device.

Bare-metal artifacts are written to
`build/baremetal/sentence_gemmini/` as
`GEMMINI_nn_gemmini_baremetal.{elf,bin,dump,map}`. The board runtime uses UART
MMIO and does not provide Spike `tohost/fromhost`; use the Linux/pk executable
above for functional Spike simulation.

The native/per-tensor path is verified on both Spike/pk and Gemmini hardware
with final INT8 logit `23` and a correct classification. The recorded
bare-metal single-sample run produced:

```text
total_cycles:          1,682,812
Gemmini total cycles:  1,682,224
host postprocess:            113
clock:                  50 MHz
latency:               0.033656 s
```

That hardware run reported `native_conv=1`, Gemmini per-tensor store-scale
requantization, fused ReLU, staged Gemmini pooling, and Gemmini FC.
