# RISC-V Inference

This repository provides a C-based neural-network inference stack targeting
RISC-V scalar CPUs, the RISC-V Vector Extension (RVV), and the Gemmini matrix
accelerator. It includes Linux/Spike-pk and bare-metal execution paths,
testbenches, and Python utilities for model preparation.

## Architecture Overview

- Model API and layer definitions live in `header/` and expose a PyTorch-like
  naming style for building neural-network graphs in C.
- Core inference code is in `csrc/`, split into layer logic, utilities, and
  backend-specific kernels (CPU and VPU/RVV).
- Testbenches in `testbench/` provide end-to-end inference runs for different
  models and data types, and include Spike performance results per test.
- Python utilities in `py/` support training, calibration, or data prep for
  specific models (e.g., sentence or ResNet variants).
- Gemmini ISA support and accelerator operators live under
  `csrc/backends/riscv/gemmini/`.

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
│       ├── vector/ops/ # Explicit RVV operators grouped by family
│       └── gemmini/ops/ # Gemmini ISA support and accelerator operators
├── header/             # Public headers and model/layer API
├── baremetal/          # Bare-metal startup, runtime, linker, and app adapters
├── testbench/          # End-to-end model testbenches + Spike results
├── py/                 # Training/calibration/data utilities
└── scripts/            # Configure, build, and checked flashing helpers
```

## CMake and Ninja Build Guide

CMake generates the build graph and Ninja performs incremental and parallel
compilation. The legacy Makefiles remain temporarily available for build
regression comparisons, but new users should use this section.

### Prerequisites

- CMake 3.16 or newer
- Ninja
- `riscv64-unknown-linux-gnu-gcc` for Linux or Spike `pk`
- `riscv64-unknown-elf-gcc` for bare-metal images

This repository may be used from a shell environment that places an older
CMake first in `PATH`. Check the selected version before configuring:

```sh
/usr/bin/cmake --version
ninja --version
```

### Load the Chipyard environment first

Before building or running the scalar CPU and Gemmini Linux/Spike programs,
load Chipyard's environment in the same shell:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh
```

If Chipyard is installed elsewhere, replace the path above with that
installation's `env.sh`. For the Gemmini `default` profile this environment
selects the RV64GC Linux compiler, Spike, `pk`, and Gemmini extension used by
the commands below. Confirm them before building:

```sh
command -v spike
command -v riscv64-unknown-linux-gnu-gcc
test -x "$RISCV/riscv64-unknown-elf/bin/pk"
```

### RVV builds use the system compiler

Do **not** source Chipyard's `env.sh` before compiling the generic RVV
profiles. Start from a shell whose `PATH` selects the separately installed RVV
Linux cross compiler:

```sh
command -v riscv64-unknown-linux-gnu-gcc
riscv64-unknown-linux-gnu-gcc --version
```

The compiler reported above must be the RVV compiler installed in the system
`PATH`, not the compiler injected by Chipyard's environment. CMake records the
compiler when a build directory is first configured, so use the backend's
separate build directory and do not reuse a directory previously configured
with another toolchain.

### Short build command

The helper configures a separate Ninja directory for every platform, backend,
testbench, and ISA profile, then builds it using all available workers:

```sh
./scripts/configure_build.sh <linux-pk|baremetal> \
    <cpu|vector|gemmini|all> <testbench> [profile]
```

Linux scalar CPU baseline:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh linux-pk cpu sentence_inference_fp32 default

spike --isa=rv64gc_zicntr_zihpm \
    pk \
    build/linux-pk/sentence_inference_fp32/default/cpu/static/sentence_inference_fp32
```

Linux/Spike explicit RVV build:

```sh
# Run this from a shell that has not sourced Chipyard env.sh.
command -v riscv64-unknown-linux-gnu-gcc

./scripts/configure_build.sh linux-pk vector sentence_inference_fp32 zvl512b_cycle
```

Bare-metal explicit RVV build:

```sh
./scripts/configure_build.sh baremetal vector sentence_inference_fp32 V128D128B
```

The profile argument is optional. The helper selects these defaults when it is
omitted:

| Platform | Backend | Default profile |
| --- | --- | --- |
| `linux-pk` | `cpu` | `default` |
| `linux-pk` | `vector` or `all` | `zvl128b` |
| `linux-pk` | `gemmini` | `default` |
| `baremetal` | `vector` | `V128D128B` |
| `baremetal` | `gemmini` | `GEMMINI` |

### ISA profiles

For Linux/Spike builds, `[profile]` becomes `NN_CONFIG` and selects the
compiler `-march` value. Use `default` for the scalar CPU baseline and a
vector-capable profile for the explicit RVV backend:

| Profile | Compiler `-march` | Intended use |
| --- | --- | --- |
| `default` | `rv64gc_zicntr_zihpm` | Scalar CPU baseline or Gemmini host code |
| `zvl64b` | `rv64gcv_zicntr_zihpm_zvbb_zvl64b_zve64d` | Generic RVV, VLEN >= 64 |
| `zvl128b` | `rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d` | Generic RVV, VLEN >= 128 |
| `zvl256b` | `rv64gcv_zicntr_zihpm_zvbb_zvl256b_zve64d` | Generic RVV, VLEN >= 256 |
| `zvl512b` | `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d` | Generic RVV, VLEN >= 512 |
| `zvl512b_cycle` | `rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d` | VLEN=512 RVV cycle benchmark |
| `RVV` | `rv64imafdcbvzicsr_zifencei_zicntr_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | Matching Chipyard RVV configuration |
| `MINV64D64RocketGENESYS2Config` | `rv64imafdcbzicsr_zifencei_zicntr_zihpm_zvl64b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | Matching Genesys2 configuration |
| `DSPV128D128RocketGENESYS2Config` | `rv64imafdcbvzicsr_zifencei_zicntr_zihpm_zvl128b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | Matching Genesys2 configuration |

The profile controls compilation; Spike must independently be started with a
compatible `--isa` string. This repository uses Spike 1.1.1-dev, which derives
VLEN and ELEN from the vector extensions in that ISA string. For example,
`zvl128b_zve64d` selects VLEN=128 and ELEN=64. This Spike version does not
provide a separate `--varch` option.

Run the `zvl128b` binary with:

```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl128b/vector/static/sentence_inference_fp32
```

The scalar CPU teaching configuration is:

```sh
# Scalar CPU: no V extension and compiler auto-vectorization is disabled.
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh \
    linux-pk cpu sentence_inference_fp32 default

spike --isa=rv64gc_zicntr_zihpm \
    "$RISCV/riscv64-unknown-elf/bin/pk" \
    build/linux-pk/sentence_inference_fp32/default/cpu/static/sentence_inference_fp32
```

The RVV teaching configuration must be built in a separate shell without
sourcing Chipyard's `env.sh`:

```sh
# Explicit RVV intrinsics with VLEN=512.
command -v riscv64-unknown-linux-gnu-gcc

./scripts/configure_build.sh \
    linux-pk vector sentence_inference_fp32 zvl512b_cycle

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl512b_cycle/vector/static/sentence_inference_fp32
```

`NN_AUTO_VECTORIZE` is forced to `OFF` by `configure_build.sh`. This keeps the
CPU baseline scalar and ensures that vector builds measure the repository's
explicit RVV intrinsic kernels rather than compiler-generated vector loops.

All CPU and RVV benchmark profiles include `zicntr_zihpm` because the
testbenches read cycle and hardware-performance counters. For bare-metal
builds, `[profile]` becomes `NN_HARDWARE_CONFIG`:

| Profile | Compiler `-march` or target | Intended target |
| --- | --- | --- |
| `V128D128B` | `rv64gcv_zicntr_zihpm_zvl128b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | RVV hardware with VLEN=128 |
| `V256D128B` | `rv64gcv_zicntr_zihpm_zvl256b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | RVV hardware with VLEN=256 |
| `V512D128B` | `rv64gcv_zicntr_zihpm_zvl512b_zve64d_zvfh_zfh_zba_zbb_zbs_zvbb` | RVV hardware with VLEN=512 |
| `GEMMINI` | Gemmini-specific RV64GC build | Gemmini bare-metal target |

The `RVV` and Genesys2 names describe specific hardware configurations. Do not
select them only because their names contain a desired VLEN; use the matching
generic `zvl*` profile for ordinary Spike experiments.

### Direct CMake commands

The helper is only a convenience wrapper. The equivalent Linux/Spike command
is:

```sh
/usr/bin/cmake -S . \
    -B build/cmake/linux-pk-sentence_inference_fp32-vector-zvl512b_cycle \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-linux-gnu.cmake \
    -DNN_PLATFORM=linux-pk \
    -DNN_BACKEND=vector \
    -DNN_TESTBENCH=sentence_inference_fp32 \
    -DNN_CONFIG=zvl512b_cycle \
    -DNN_LINK_MODE=static \
    -DNN_AUTO_VECTORIZE=OFF

/usr/bin/cmake --build \
    build/cmake/linux-pk-sentence_inference_fp32-vector-zvl512b_cycle --parallel
```

Run the generated static ELF with Spike:

```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl512b_cycle/vector/static/sentence_inference_fp32
```

The equivalent bare-metal configuration is:

```sh
/usr/bin/cmake -S . \
    -B build/cmake/baremetal-sentence-vector-V128D128B \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-baremetal.cmake \
    -DNN_PLATFORM=baremetal \
    -DNN_BACKEND=vector \
    -DNN_TESTBENCH=sentence_inference_fp32 \
    -DNN_HARDWARE_CONFIG=V128D128B

/usr/bin/cmake --build \
    build/cmake/baremetal-sentence-vector-V128D128B --parallel
```

Ninja's default bare-metal target creates both the ELF and raw binary. Generate
the optional mixed source/assembly dump with:

```sh
/usr/bin/cmake --build \
    build/cmake/baremetal-sentence-vector-V128D128B \
    --target baremetal-dump --parallel
```

Generated files keep the public output layout independent of the internal
Ninja directory:

```text
build/linux-pk/<testbench>/<config>/<backend>/<link-mode>/<testbench>
build/baremetal/<testbench>/<hardware>_nn_rvv_baremetal.elf
build/baremetal/<testbench>/<hardware>_nn_rvv_baremetal.bin
build/baremetal/<testbench>/<hardware>_nn_rvv_baremetal.map
build/baremetal/<testbench>/<hardware>_nn_rvv_baremetal.dump
```

### Supported build selections

Linux testbench names are:

```text
gesture_recognition_fp32
kyber_nouv_rvv
resnet50
sentence_inference_fp32
sentence_inference_int8
```

`resnet50` additionally requires the generated, git-ignored
`testbench/resnet50/resnet50_weights.h`. Generate the model files using the
tools under `py/resnet50/` before configuring that target; CMake stops during
configuration with a clear error when the header is absent.

Linux profiles are `default`, `zvl64b`, `zvl128b`, `zvl256b`, `zvl512b`,
`zvl512b_cycle`, `RVV`, `MINV64D64RocketGENESYS2Config`, and
`DSPV128D128RocketGENESYS2Config`. Use `default` for the scalar CPU baseline.
Compiler auto-vectorization is disabled unless CMake is configured with
`-DNN_AUTO_VECTORIZE=ON`; explicit RVV intrinsics remain enabled.

The Gemmini backend defaults to the latest native/per-tensor SentenceCNN path.
It uses native `tiled_conv_auto`, Gemmini store-scale requantization and ReLU,
staged Gemmini pooling, and Gemmini fully connected layers. The earlier
per-channel im2col path remains as a regression fallback. Operator contracts
and current limitations are documented in
[`csrc/backends/riscv/gemmini/ops/README.md`](csrc/backends/riscv/gemmini/ops/README.md).

Build and run the imported Gemmini SentenceCNN on Spike `pk` using the
Chipyard toolchain and extension plugin:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh \
    linux-pk gemmini sentence_gemmini default

/usr/bin/cmake --build \
    build/cmake/linux-pk-sentence_gemmini-gemmini-default \
    --target run-spike-gemmini
```

The equivalent direct execution command is:

```sh
spike --isa=rv64gc_zicntr_zihpm --extension=gemmini \
    "$RISCV/riscv64-unknown-elf/bin/pk" \
    build/linux-pk/sentence_gemmini/default/gemmini/static/sentence_gemmini
```

Build and simulate the bit-exact scaled INT8 FFT with the same environment:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh \
    linux-pk gemmini fft_batched_int8_gemmini default

spike --isa=rv64gc_zicntr_zihpm --extension=gemmini \
    "$RISCV/riscv64-unknown-elf/bin/pk" \
    build/linux-pk/fft_batched_int8_gemmini/default/gemmini/static/fft_batched_int8_gemmini
```

The FFT target combines eight complex twiddle rotations in each 16x16 Gemmini
tile, processes 64 batches per matrix multiplication, runs 10 iterations, and
validates all 65,536 complex points against the shared Q7 ground truth. The
current RV64GC Gemmini Spike reference passes bit-exact validation with zero
mismatches and averages 15,288,837 cycles for the complete FFT region. This
includes bit reversal, matrix packing, Gemmini execution/readback, and scalar
Q7 post-processing, but excludes input reload and one-time construction of the
513 prepacked Gemmini twiddle tiles.
The block-diagonal butterfly mapping, Q7 correction, and host/accelerator split
are documented in
[`testbench/fft_batched_int8_gemmini/README.md`](testbench/fft_batched_int8_gemmini/README.md).
The consolidated CPU/RVV/Gemmini FP32 and INT8 hardware comparison, including
cycles, milliseconds, fusion effects, and speedup ratios, is maintained in
[`testbench/fft/README.md`](testbench/fft/README.md#batched-fft-cross-architecture-comparison).

The current native/per-tensor reference produces INT8 logit `23` and a correct
classification on both Spike/pk and Gemmini hardware. The recorded hardware
run takes `1,682,812` cycles at 50 MHz.

Bare-metal adapters are `sentence_inference_fp32`, `sentence_inference_int8`,
`gesture_model`, `kyber`, `sentence_gemmini`, and
`fft_batched_int8_gemmini`. Vector targets use their VLEN/datapath profile;
Gemmini targets use the `GEMMINI` RV64GC profile.

Build the Gemmini bare-metal image with the main repository runtime:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh \
    baremetal gemmini sentence_gemmini GEMMINI
```

This configuration records the reference Makefile requirements in CMake:
RV64GC/LP64D, medany, freestanding compilation, Gemmini preallocation, no
startup libraries, the common `baremetal/linker.ld`, split minilib sources,
syscalls, trap handler, and startup assembly. Generated files are:

```text
build/baremetal/sentence_gemmini/GEMMINI_nn_gemmini_baremetal.elf
build/baremetal/sentence_gemmini/GEMMINI_nn_gemmini_baremetal.bin
build/baremetal/sentence_gemmini/GEMMINI_nn_gemmini_baremetal.map
```

Generate the disassembly with:

```sh
/usr/bin/cmake --build \
    build/cmake/baremetal-sentence_gemmini-gemmini-GEMMINI \
    --target baremetal-dump
```

The FFT Gemmini bare-metal equivalent is:

```sh
./scripts/configure_build.sh \
    baremetal gemmini fft_batched_int8_gemmini GEMMINI

/usr/bin/cmake --build \
    build/cmake/baremetal-fft_batched_int8_gemmini-gemmini-GEMMINI \
    --target baremetal-dump
```

Its ELF, BIN, dump, and map are under
`build/baremetal/fft_batched_int8_gemmini/`.

The bare-metal ELF uses the board UART MMIO console and hardware boot ABI. It
can be loaded by Spike for instruction-level debugging, but standard Spike
reports that `tohost`/`fromhost` are absent and cannot display the board UART or
observe normal program termination. Use the Linux/`pk` target above for the
current automated Gemmini Spike test:

```sh
./scripts/configure_build.sh linux-pk gemmini sentence_gemmini default
/usr/bin/cmake --build \
    build/cmake/linux-pk-sentence_gemmini-gemmini-default \
    --target run-spike-gemmini
```

A future HTIF console/exit adapter may enable a separate Spike-oriented
bare-metal configuration without changing the flashable hardware runtime.

To use a non-default compiler installation, set the prefix while creating a
fresh build directory:

```sh
/usr/bin/cmake -S . -B build/cmake/custom-linux -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv-linux-gnu.cmake \
    -DRISCV_LINUX_PREFIX=/opt/riscv-linux/bin/riscv64-unknown-linux-gnu- \
    -DNN_PLATFORM=linux-pk -DNN_BACKEND=vector \
    -DNN_TESTBENCH=sentence_inference_fp32 -DNN_CONFIG=zvl128b
```

The bare-metal equivalent variable is `RISCV_BAREMETAL_PREFIX`.

### Cleaning

Remove one configuration without touching other builds:

```sh
/usr/bin/cmake --build <ninja-build-directory> --target clean
```

This removes files known to that Ninja configuration. Public artifacts from
other configurations intentionally remain under `build/`. To remove every
generated configuration and artifact, remove the repository's `build/`
directory after confirming that it contains no manually stored data.

### SD-card flashing

First build and inspect the raw `.bin`. Then reconfigure the same bare-metal
Ninja directory with the whole SD-card device. Omitting confirmation performs
the safety check and intentionally returns status 2 without writing:

```sh
/usr/bin/cmake -S . \
    -B build/cmake/baremetal-sentence-vector-V128D128B \
    -DNN_SDCARD_DEVICE=/dev/sdc

/usr/bin/cmake --build \
    build/cmake/baremetal-sentence-vector-V128D128B \
    --target baremetal-flash
```

After checking the printed model, type, size, and empty mount point, authorize
the destructive write and run the target again:

```sh
/usr/bin/cmake -S . \
    -B build/cmake/baremetal-sentence-vector-V128D128B \
    -DNN_SDCARD_DEVICE=/dev/sdc \
    -DNN_FLASH_CONFIRM=YES

/usr/bin/cmake --build \
    build/cmake/baremetal-sentence-vector-V128D128B \
    --target baremetal-flash
```

The default destination is 512-byte block 34. Override it only when the boot
flow requires another location with `-DNN_SDCARD_BLOCK=<block>`. Selecting the
wrong device destroys data.

## Legacy Make Build Guide

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
    CONFIG=zvl512b_cycle \
    LINK_MODE=static

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl512b_cycle/vector/static/sentence_inference_fp32
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
given. Architecture profiles include `default` (`rv64gc_zicntr_zihpm`), `zvl64b`,
`zvl128b`, `zvl256b`, `zvl512b`, `zvl512b_cycle`, and the board-specific
profiles defined in `makefile`. Use `CONFIG=default` with `BACKEND=cpu`; vector
and `all` builds require a vector-capable profile.

`LINK_MODE=static` is the default and is appropriate for Spike `pk`:

```sh
make linux TESTBENCH=sentence_inference_fp32 BACKEND=vector \
    CONFIG=zvl512b_cycle LINK_MODE=static

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl512b_zve64d \
    pk \
    build/linux-pk/sentence_inference_fp32/zvl512b_cycle/vector/static/sentence_inference_fp32
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

## Supported Architectures and Chipyard

The repository is designed to work with SoCs generated by the open-source
Chipyard project. Chipyard supplies configurable Rocket/BOOM-class RISC-V
cores, the Gemmini accelerator integration, simulation infrastructure, and
FPGA prototyping flows. This repository supplies the neural-network operators,
model testbenches, hosted executables, and flashable bare-metal programs that
run on those systems.

| Backend | Chipyard hardware | Execution environments | Notes |
| --- | --- | --- | --- |
| `cpu` | 64-bit scalar RISC-V core | Linux, Spike/pk, bare-metal runtime | Reference and baseline kernels; RVV auto-vectorization is disabled by default |
| `vector` | RISC-V core configured with RVV | Linux, Spike/pk, FPGA bare-metal | Explicit RVV intrinsics; current profiles cover minimum VLEN values from 64 to 512 bits |
| `gemmini` | Chipyard SoC with Gemmini RoCC accelerator | Gemmini-enabled Spike/pk, FPGA bare-metal | Native INT8 Conv1D, fused requantization/ReLU, staged pooling, and FC |

The current ABI is RV64 `lp64d`. Linux/pk builds use a
`riscv64-unknown-linux-gnu` toolchain, while flashable programs use a
`riscv64-unknown-elf` bare-metal toolchain. The supplied configuration names,
such as `MINV64D64RocketGENESYS2Config`,
`DSPV128D128RocketGENESYS2Config`, `V128D128B`, `V256D128B`, and
`V512D128B`, describe the profiles already wired into this repository.

### How this repository fits into Chipyard

```text
Chipyard SoC configuration
        |
        +-- scalar RISC-V core ------> BACKEND=cpu
        +-- RVV-enabled core --------> BACKEND=vector
        +-- Gemmini RoCC accelerator -> BACKEND=gemmini
        |
        +-- Spike/pk simulation or FPGA bitstream
                          |
                          +-- executable from build/linux-pk/
                          +-- image from build/baremetal/
```

For functional Gemmini simulation, source the Chipyard environment before
running the repository target:

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

# Select sentence_gemmini or fft_batched_int8_gemmini.
GEMMINI_TESTBENCH=fft_batched_int8_gemmini
./scripts/configure_build.sh \
    linux-pk gemmini "$GEMMINI_TESTBENCH" default
/usr/bin/cmake --build \
    "build/cmake/linux-pk-${GEMMINI_TESTBENCH}-gemmini-default" \
    --target run-spike-gemmini
```

For FPGA execution, first generate and program a compatible Chipyard bitstream,
then build this repository with the matching backend/profile. The boot ROM,
memory map, UART address, ISA extensions, vector length, Gemmini parameters,
clock frequency, and SD-card load offset must agree with the selected Chipyard
configuration. A profile name alone cannot make an incompatible bitstream run
the program.

The source also contains scalar reference paths used for correctness checks.
Custom VPU performance counters and privileged CSRs depend on the execution
environment and are not assumed to be available in Linux user mode.

## Operator Coverage

| Operator family | Data types | Implementations |
| --- | --- | --- |
| Conv1D | INT8, FP32 | Scalar reference, RVV im2col, and Gemmini native INT8 kernels |
| Conv2D | INT8, FP32 | Scalar reference and RVV im2col kernels |
| Fully connected | INT8, FP32 | Scalar reference, RVV, and Gemmini INT8 kernels |
| Pooling | INT8, FP32 | 1D/2D scalar and RVV kernels; staged Gemmini INT8 pooling |
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
toolchains and vector intrinsics, the Chipyard SoC-generation ecosystem,
Gemmini, and the Spike ISA simulator with the proxy kernel for hosted RISC-V
testing. The model-building interface and test flow are inspired by common
neural-network framework conventions.

## License

The project-owned source code is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the complete terms. Third-party components retain
their original licenses; in particular, the headers under
`csrc/backends/riscv/gemmini/ops/include/` retain their Gemmini/RoCC copyright and
license notices.

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
