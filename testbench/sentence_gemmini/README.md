# Gemmini SentenceCNN testbench

Copyright 2026 Stanley Lee
SPDX-License-Identifier: Apache-2.0

This testbench runs the INT8 SentenceCNN through the default 16x16 Gemmini
configuration. The same inference source is compiled in two environments:

- Linux ABI with Spike `pk` and `--extension=gemmini`;
- freestanding bare-metal with the repository startup, linker, minilib,
  syscalls, and trap runtime.

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
source /home/mc2/chipyard_1.13.0/chipyard/env.sh
./scripts/configure_build.sh linux-pk gemmini sentence_gemmini default
/usr/bin/cmake --build \
    build/cmake/linux-pk-sentence_gemmini-gemmini-default \
    --target run-spike-gemmini
```

Build the flashable bare-metal image:

```sh
./scripts/configure_build.sh baremetal gemmini sentence_gemmini GEMMINI
```

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
