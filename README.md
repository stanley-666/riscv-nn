# RISC-V 1D CNN Inference

This repository provides a C-based 1D CNN inference stack targeting RISC-V,
with both scalar CPU and vector (RVV/VPU) execution paths, plus testbenches
and Python utilities for model preparation.

## Architecture Overview

- Model API and layer definitions live in `header/` and expose a PyTorch-like
  naming style for building 1D CNN graphs in C.
- Core inference code is in `csrc/`, split into layer logic, utilities, and
  backend-specific kernels (CPU and VPU/RVV).
- Testbenches in `testbench/` provide end-to-end inference runs for different
  models and data types, and include Spike performance results per test.
- Python utilities in `py/` support training, calibration, or data prep for
  specific models (e.g., sentence or ResNet variants).
- `gemmini/` contains Gemmini integration and related software for accelerator
  experiments.

### Public and Internal Headers

Applications should include the public umbrella header:

```c
#include "riscv_nn.h"
```

`header/nn_ops.h` contains the stable operator entry points selected by the
inference dispatcher. Architecture experiments such as LMUL, unroll,
accumulator, chaining, and load-scheduling variants are declared in
`csrc/backends/riscv/vector/ops/<family>/*_internal.h`. Those internal headers remain available to
repository benchmarks but are not part of the stable public API.

## Repository Layout

```
.
├── csrc/               # Core inference and runtime source
│   └── backends/riscv/
│       ├── cpu/ops/    # CPU/reference operators
│       └── vector/ops/ # Explicit RVV operators grouped by family
├── header/             # Public headers and model/layer API
├── baremetal/          # Bare-metal startup, runtime, linker, and app adapters
├── testbench/          # End-to-end model testbenches + Spike results
├── py/                 # Training/calibration/data utilities
└── gemmini/            # Gemmini integration and tooling
```

## Complete Build Guide

### Quick Start

Run every command from the repository root. Choose one of the following flows;
do not run a bare-metal ELF with `pk`, and do not flash a Linux ELF to an SD
card.

Linux/Spike pk with the scalar CPU backend:

```sh
make -j$(nproc) linux \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=cpu \
    CONFIG=default \
    LINK_MODE=static
```

Linux/Spike pk with explicit RVV and a 128-bit VLEN profile:

```sh
make -j$(nproc) linux \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    CONFIG=zvl128b \
    LINK_MODE=static

spike --isa=rv64gcv_zvbb_zvl128b_zve64d \
    --varch=vlen:128,elen:64 \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl128b/vector/static/sentence_inference_fp32
```

Bare-metal ELF and raw SD-card image for the `V128D128B` hardware profile:

```sh
make -j$(nproc) baremetal \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

Generated files:

```text
build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.elf
build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.bin
build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.map
```

The common command form is:

```sh
make -j$(nproc) <mode> TESTBENCH=<name> BACKEND=<cpu|vector|all> [configuration]
```

`-j$(nproc)` lets Make compile independent source files concurrently using the
available CPU threads. Omit it when a serial build is required. Both Linux and
bare-metal builds compile one object per source file, so incremental and
parallel builds are supported.

### What Is Shared Between Platforms?

Linux/Spike and bare-metal intentionally use different startup and runtime
implementations, but share the testbench model graph, weights, inference
dispatcher, and operator sources:

```text
Linux loader / Spike pk                 baremetal/start.S
          |                                     |
Linux testbench main                    baremetal/main.c
          |                                     |
          +------ shared sentence model --------+
                         |
                 forward_fp32_vpu()
                         |
       csrc/backends/riscv/vector/ops/*
```

The common sentence FP32 graph is defined once in
`testbench/sentence_inference_fp32/sentence_model.h`. Both environments execute:

```text
Conv1D -> Conv1D -> Conv1D -> AdaptiveMaxPool1D -> FC/ReLU -> FC/Sigmoid
```

The platform-specific boundary is:

| Component | Linux / Spike pk | Bare-metal |
| --- | --- | --- |
| Program entry | `testbench/sentence_inference_fp32/sentence_inference_fp32.c` | `baremetal/start.S` then `baremetal/main.c` |
| Allocation runtime | `csrc/nn_runtime_linux.c` | `baremetal/nn_runtime_baremetal.c` and the 64-byte-aligned minilib allocator |
| Timing | Hosted C runtime and `rdcycle` where available | Platform timer and hardware cycle counter |
| Link environment | Linux ABI, static for `pk` or dynamic for Linux | Freestanding linker script at `0x80000000` |

Changing the platform runtime does not select a different neural-network
operator. `BACKEND=vector` selects the same explicit RVV sources for both
environments.

### Default Inference Benchmark Output

The sentence FP32 testbench runs one input by default on both Linux/Spike and
bare-metal. It records each operator's cycle count and the complete forward-pass
cycle count, then prints the report only after inference has finished:

```text
Layer 0 (CONV1D): ... cycles
Layer 1 (CONV1D): ... cycles
Layer 2 (CONV1D): ... cycles
Layer 3 (POOL1D): ... cycles
Layer 4 (FC): ... cycles
Layer 5 (FC): ... cycles
Total inference: ... cycles
```

No reporting `printf` call is executed inside a measured layer interval. The
total interval covers input-buffer preparation performed by the dispatcher,
all six layers, buffer swaps, and dispatcher overhead. Model construction,
buffer allocation, result printing, and accuracy messages are outside the total
inference interval. The full dataset callback remains available in the source
for accuracy testing but is not part of the default benchmark entry.

`<mode>` selects the runtime and toolchain, while `BACKEND` selects the operator
sources compiled into the program:

| Option | Meaning |
| --- | --- |
| `linux` | Linux or Spike `pk`; uses `riscv64-unknown-linux-gnu-gcc` and `csrc/nn_runtime_linux.c` |
| `baremetal` | Freestanding image; uses `riscv64-unknown-elf-gcc` and the runtime under `baremetal/` |
| `BACKEND=cpu` | Compile only scalar CPU operators and `forward()` |
| `BACKEND=vector` | Compile only explicit RVV operators and the VPU inference dispatcher |
| `BACKEND=all` | Compile both implementations; the testbench chooses which dispatcher it calls |

### Prerequisites

Install or provide the appropriate cross compiler:

```text
Linux / Spike pk: riscv64-unknown-linux-gnu-gcc
Bare-metal:       riscv64-unknown-elf-gcc
```

Override the compiler prefix when it is not on `PATH`:

```sh
make linux CROSS_COMPILE=/opt/riscv-linux/bin/riscv64-unknown-linux-gnu- ...
make baremetal CROSS_COMPILE=/opt/riscv-elf/bin/riscv64-unknown-elf- ...
```

### Linux and Spike pk

Build the scalar CPU baseline without the Vector Extension:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=cpu CONFIG=default
make linux TESTBENCH=sentence_inference_int8 BACKEND=cpu CONFIG=default
```

Build the explicit RVV implementation:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=vector CONFIG=zvl128b
make linux TESTBENCH=sentence_inference_int8 BACKEND=vector CONFIG=zvl512b_cycle
```

Build both backends into the same testbench binary:

```sh
make linux TESTBENCH=sentence_inference_int8 BACKEND=all CONFIG=zvl512b
```

The available Linux testbenches are printed when an invalid `TESTBENCH` is
given. Architecture profiles include `default` (`rv64gc`), `zvl64b`,
`zvl128b`, `zvl256b`, `zvl512b`, `zvl512b_cycle`, and the board-specific
profiles defined in `makefile`. Use `CONFIG=default` with `BACKEND=cpu`; vector
and `all` builds require a vector-capable profile.

`LINK_MODE=static` is the default and is appropriate for Spike `pk`:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=vector \
    CONFIG=zvl128b LINK_MODE=static

spike --isa=rv64gcv pk \
    build/linux-pk/sentence_inference_fp32/zvl128b/vector/static/sentence_inference_fp32
```

For a normal RISC-V Linux userspace with a dynamic loader, request a dynamic
binary:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=cpu \
    CONFIG=default LINK_MODE=dynamic
```

Linux/pk outputs are ELF executables even though their filenames have no
`.elf` suffix. The output layout records every build choice:

```text
build/linux-pk/<testbench>/<config>/<backend>/<link-mode>/<testbench>
```

The shorter direct targets remain available, for example:

```sh
make sentence_inference_int8_static BACKEND=vector CONFIG=zvl512b
make sentence_inference_int8_dynamic BACKEND=cpu CONFIG=default
```

### Bare-metal

The bare-metal application adapters currently support
`sentence_inference_fp32`, `sentence_inference_int8`, `gesture_model`, and
`kyber`. They use the same RVV operators as the Linux / Spike pk builds. Build
a raw image, ELF, or disassembly with:

```sh
make baremetal TESTBENCH=sentence_inference_fp32 BACKEND=vector
make baremetal-elf TESTBENCH=sentence_inference_fp32 BACKEND=vector
make baremetal-dump TESTBENCH=sentence_inference_fp32 BACKEND=vector
```

Build the INT8 sentence benchmark with:

```sh
make -j$(nproc) baremetal \
    TESTBENCH=sentence_inference_int8 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

The INT8 adapter uses five warm-up runs followed by 100 measured inference
runs. It reports the average cycle count using only one cycle-counter read
before and after the measured loop. Its ping-pong buffers are statically
allocated and aligned to 64 bytes.

Build the Gesture FP32 and Kyber images with:

```sh
make -j$(nproc) baremetal TESTBENCH=gesture_model BACKEND=vector \
    HARDWARE_CONFIG=V128D128B

make -j$(nproc) baremetal TESTBENCH=kyber BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

Their raw images are written to:

```text
build/baremetal/gesture_model/V128D128B_nn_rvv_baremetal.bin
build/baremetal/kyber/V128D128B_nn_rvv_baremetal.bin
```

Use the same testbench name with `baremetal-flash`. For example:

```sh
make baremetal-flash TESTBENCH=gesture_model BACKEND=vector \
    HARDWARE_CONFIG=V128D128B SDCARD_DEVICE=/dev/sdc FLASH_CONFIRM=YES

make baremetal-flash TESTBENCH=kyber BACKEND=vector \
    HARDWARE_CONFIG=V128D128B SDCARD_DEVICE=/dev/sdc FLASH_CONFIRM=YES
```

Select the hardware vector profile when required:

```sh
make baremetal TESTBENCH=sentence_inference_fp32 BACKEND=vector \
    HARDWARE_CONFIG=V512D128B
```

Supported bare-metal hardware profiles are `V128D128B`, `V256D128B`, and
`V512D128B`. Products are written below:

```text
build/baremetal/<testbench>/<hardware>_nn_rvv_baremetal.{elf,bin,dump,map}
```

The bare-metal build shares the model, layers, and operator sources in
`csrc/`, but replaces Linux allocation and system support with
`baremetal/nn_runtime_baremetal.c`, the 64-byte-aligned allocator, startup,
minilib, trap handling, and linker script. Each adapter reuses model data from
its matching directory under `testbench/`.

### Build controls and cleanup

Compiler auto-vectorization is disabled by default for every backend so the CPU
baseline stays scalar even when `-march` contains `v`. Explicit RVV intrinsics
are unaffected. Enable compiler-generated vectorization only for experiments:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=vector \
    CONFIG=zvl128b AUTO_VECTORIZE=1
```

Each backend uses a separate object directory, so changing `BACKEND` does not
reuse objects from another implementation. All generated objects, ELF files,
bare-metal images, disassemblies, and linker maps are kept under `build/`.
Remove every generated build product for both platforms with:

```sh
make clean
```

### Staged Bare-metal Build and SD-card Installation

Use the following stages when preparing a program for the board. Keeping the
build and flash operations separate makes it possible to inspect every artifact
before touching the SD card.

#### Stage 1: start from a clean build tree

```sh
make clean
```

This removes all generated Linux, Spike pk, and bare-metal files under
`build/`. It does not remove source code or model data.

#### Stage 2: compile and link the bare-metal ELF

```sh
make -j$(nproc) baremetal-elf \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

The ELF contains section addresses, symbols, and debug information. The linker
places the program at the RAM address defined by `baremetal/linker.ld`, which is
currently `0x80000000`:

```text
build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.elf
```

#### Stage 3: inspect the ELF and optionally generate a disassembly

```sh
make -j$(nproc) baremetal-dump \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B

riscv64-unknown-elf-readelf -h \
    build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.elf
```

The disassembly is written beside the ELF with a `.dump` suffix.

#### Stage 4: generate the raw boot image

```sh
make -j$(nproc) baremetal \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

This keeps the ELF and generates the raw image consumed by the board boot flow:

```text
build/baremetal/sentence_inference_fp32/V128D128B_nn_rvv_baremetal.bin
```

The `.elf` is for inspection and debugging. Write the `.bin` to the SD card.

#### Stage 5: identify and unmount the SD card

```sh
lsblk -o NAME,PATH,MODEL,TRAN,RM,TYPE,SIZE,MOUNTPOINT
```

The device must be the whole disk, such as `/dev/sdc`, not a partition such as
`/dev/sdc1`. Unmount every partition before continuing. Device names can change
after reconnecting hardware, so check `lsblk` every time.

#### Stage 6: run the non-writing flash safety check

For an SD card detected as `/dev/sdc`:

```sh
make baremetal-flash \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdc
```

This command validates that the path exists, is a whole block device, and has no
mounted partitions. Without `FLASH_CONFIRM=YES`, it prints the selected device
and stops without compiling or writing. A successful non-writing check looks
similar to:

```text
Flash check passed for /dev/sdc.
NAME MODEL          TYPE   SIZE MOUNTPOINT
sdc  Storage_Device disk 234.2G
Re-run with FLASH_CONFIRM=YES to write ...bin at block 34.
make: *** [baremetal-flash] Error 2
```

The final `Error 2` is intentional in this stage. It makes the safety-only
command return a non-success status so it cannot silently continue into a flash
operation. The important success message is `Flash check passed`, together with
the expected device model, type, size, and an empty mount point.

#### Stage 7: flash the program to the SD card

Only after confirming that `/dev/sdc` is the intended SD card, explicitly
authorize the write:

```sh
make baremetal-flash \
    TESTBENCH=sentence_inference_fp32 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdc \
    FLASH_CONFIRM=YES
```

The target performs the safety checks again, builds the `.bin` when necessary,
and writes it using a 512-byte block size at block 34:

```text
SD-card byte offset = 34 * 512 = 17408 bytes
```

`SDCARD_BLOCK=34` is inherited from the current board boot flow. Override it
only if the boot ROM configuration uses another block. Selecting the wrong
device or block can permanently destroy data. Do not remove the SD card while
`dd` is running; wait until the command prints `Flash completed.` and returns to
the shell prompt.

## Supported Architectures

The optimized kernels target 64-bit RISC-V systems implementing the RISC-V
Vector Extension. The current build profiles use the `lp64d` ABI and cover
minimum vector lengths from 64 to 512 bits.

| Environment | Toolchain target | Configured vector profiles |
| --- | --- | --- |
| Linux and Spike pk | `riscv64-unknown-linux-gnu` | Zvl128b, Zvl256b, Zvl512b |
| Bare-metal | `riscv64-unknown-elf` | Zvl128b, Zvl256b, Zvl512b |

The source also contains scalar reference paths used for correctness checks.
Custom VPU performance counters and privileged CSRs depend on the execution
environment and are not assumed to be available in Linux user mode.

## Operator Coverage

| Operator family | Data types | Implementations |
| --- | --- | --- |
| Conv1D | INT8, FP32 | Scalar reference and RVV im2col kernels |
| Conv2D | INT8, FP32 | Scalar reference and RVV im2col kernels |
| Fully connected | INT8, FP32 | Scalar reference and RVV kernels |
| Pooling | INT8, FP32 | 1D and 2D scalar/RVV kernels |
| Activation | INT8, FP32 | ReLU, leaky ReLU, softmax, and supporting post-processing |
| Transformer | FP32 | Layer normalization and attention support |
| Tensor utilities | INT8, FP32 | Transpose, add, save, and layout helpers |

Conv1D includes multiple LMUL, unroll, accumulator, and load-scheduling
variants for architecture benchmarking. These experimental entry points may
change. The default FP32 RVV inference dispatcher uses the im2col LMUL8,
unroll-8, two-accumulator implementation:

```text
conv1d_fp32_vpu_im2col_unroll8_acc2_m8
```

## Acknowledgements

This project is developed by MC2 Lab at National Taiwan Normal University.
It builds on the RISC-V ISA and Vector Extension ecosystem, the GNU RISC-V
toolchains and vector intrinsics, and the Spike ISA simulator with the proxy
kernel for hosted RISC-V testing. The model-building interface and test flow
are inspired by common neural-network framework conventions.

## License

The project-owned source code is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the complete terms. Third-party components retain
their original licenses; in particular, `gemmini/rocc-software/` includes its
own license and copyright notices.

## Testbench Notes

Each testbench directory includes its own `README.md` with Spike performance
tables. Use those files to record metrics such as cycles, CPI, and runtime.

## Implementation Notes

- Compiler auto-vectorization and builtin expansion are disabled by default on
  both platforms. RVV kernels use explicit intrinsics, while scalar/reference
  kernels remain true scalar baselines even when the selected `-march` includes
  `V`; calls such as `memcpy` are not silently expanded into RVV loops. Set
  `AUTO_VECTORIZE=1` only when intentionally evaluating compiler-generated
  vector code.
- The inference runtime uses ping-pong working buffers (`buffer1`, `buffer2`).
  The final output buffer depends on layer-count parity: odd layer counts end
  in `buffer2`, even layer counts end in `buffer1`.

Numerical algorithms, approximation ranges, quantization contracts, expected
errors, and operator-specific validation procedures are documented beside the
operator sources. For activation details, see
[`csrc/backends/riscv/vector/ops/activation/README.md`](csrc/backends/riscv/vector/ops/activation/README.md).
