<!-- SPDX-FileContributor: Person: Stanley Lee -->
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
└── include/                    # Platform and device definitions
```

Bare-metal targets compile the same source file and `main()` used by the
Linux/Spike target under `testbench/`. Only startup, linking, allocation,
console output, and other runtime services are replaced.

Before entering the shared testbench `main()`, `start.S` calls
`baremetal_print_hardware_info()`. Every image therefore prints the testbench
name, build profile, configured CPU clock, machine identification CSRs, XLEN/FLEN,
and—for a vector-ISA build running on hardware that advertises RVV—its VLEN at
the top of the UART log. Pure CPU builds do not access the V-only `vlenb` CSR;
unavailable optional fields are omitted.

Both the raw `misa` value and a canonical instruction-extension string such as
`rv64imafdcbv` are printed. Privilege modes (`m/s/u`) and hypervisor support
are reported separately to avoid confusing them with instruction extensions.
When `misa.B=1`, the banner identifies the complete `Zba/Zbb/Zbs` B bundle.
When it is zero, individual `Zb*` support remains unknown until representative
instructions are probed; `Zvbb` likewise requires its own vector-instruction
probe.
The displayed build profile (for example, `V128D128B`) and configured CPU
clock come from build arguments; they are not claimed as runtime-detected
hardware values. Trap-only state is not included in this startup banner: if a
fault occurs, `trap_handler` reports `mcause`, `mepc`, `mstatus`, and `mtval`
at the point of failure.

Extension reporting is diagnostic only. Missing, unadvertised, or
probe-required extensions are printed as such, but startup always continues
into the shared testbench `main()`; the banner never rejects an image.

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

Keep the complete benchmark, including `main()`, in its existing `testbench/`
directory. Use `nn_runtime.h` for allocation and cycle timing, then register
the testbench source path in `cmake/Baremetal.cmake` and
`baremetal/Makefile`. No second application or task adapter is needed.

```sh
make -f makefile baremetal TESTBENCH=your_testbench
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
