# Bare-Metal Build

This directory contains the platform-specific runtime required to build the
shared neural-network library without Linux or Spike proxy-kernel services.
The kernels, layers, activations, and public APIs are compiled directly from
the repository-level `csrc/` and `header/` directories.

## Runtime Selection

Both platforms expose the same allocation API through `header/nn_runtime.h`:

- Linux and Spike pk link `csrc/nn_runtime_linux.c`.
- Bare-metal builds link `baremetal/nn_runtime_baremetal.c`.

The bare-metal allocator uses the 64-byte-aligned `malloc` and `free`
implementation in `minilib/alloc.c`. The bare-metal Makefile explicitly excludes
`csrc/nn_runtime_linux.c`, so the two runtime implementations are never linked
into the same binary.

## Directory Layout

```text
baremetal/
├── Makefile
├── start.S                     # Reset entry and machine-state setup
├── linker.ld                   # Bare-metal memory layout
├── minilib/                    # Independently maintained libc modules
│   ├── alloc.c                 # 64-byte-aligned heap allocator
│   ├── console.c               # UART-backed printf
│   ├── math.c                  # Minimal floating-point helpers
│   ├── memory.c                # memcpy, memset, and memcmp
│   └── process.c               # Process termination shim
├── syscalls.c                  # Platform syscall stubs
├── trap.c                      # Trap reporting
├── nn_runtime_baremetal.c      # Shared nn_runtime API implementation
├── include/                    # Platform and device definitions
└── apps/sentence_fp32/         # Default bare-metal benchmark adapter
```

The default sentence benchmark reuses weights and input data from
`testbench/sentence_inference_fp32/`; it does not keep a second copy of the
model data.

## Building

Run these commands from the repository root:

```sh
# Build the default RVV bare-metal ELF and binary image.
make -f makefile baremetal TESTBENCH=sentence_inference_fp32

# Build only the ELF.
make -f makefile baremetal-elf

# Generate a disassembly after building the ELF.
make -f makefile baremetal-dump
```

The default configuration targets a 128-bit minimum vector length. Select a
different configuration when required:

```sh
make -f makefile baremetal HARDWARE_CONFIG=V256D128B
make -f makefile baremetal HARDWARE_CONFIG=V512D128B
```

The compiler prefix and CPU frequency are configurable:

```sh
make -f makefile baremetal \
  CROSS_COMPILE=/path/to/riscv64-unknown-elf- \
  BAREMETAL_CPU_HZ=50000000UL
```

Build products are written below a testbench-specific directory, for example
`build/baremetal/sentence_inference_fp32/`.

## Adding a Bare-Metal Benchmark

Create an application directory containing its model builder, task adapter,
and demo entry points. Model data can remain in an existing testbench
directory. Register the testbench-to-adapter mapping in `baremetal/Makefile`.
It can then use the same root-level interface:

```sh
make -f makefile baremetal TESTBENCH=your_testbench
```

For development, custom paths can still be passed directly to the sub-make:

```sh
make -C baremetal vector \
  APP_DIR=/absolute/path/to/app \
  MODEL_DATA_DIR=/absolute/path/to/model/data
```

Application code should call `safe_malloc`, `safe_calloc`, and `safe_free`
instead of directly selecting a platform allocator. Code inside the shared
`csrc/` directory must remain independent of the Linux and bare-metal runtime
implementations.

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.

SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/
