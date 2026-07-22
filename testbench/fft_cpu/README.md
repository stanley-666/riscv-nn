# Scalar CPU FFT testbench

This directory contains a scalar CPU batch of 64 independent, single-precision
radix-2 FFTs. `fft_cpu.c` contains no RVV intrinsics and reads its input and
NumPy ground truth from the local `fft_vectors.h`. This header is generated with
the same size and generator as the RVV FFT vectors, allowing direct comparison.

The CPU keeps its natural `[batch][bin]` layout and executes one FFT per batch,
so every scalar FFT traverses contiguous bins. The runtime plan uses the same
four-buffer design and timing boundaries as `fft_batched`: in-place data real
and imaginary buffers plus real and imaginary twiddle buffers. All buffers are
sized from $N$ and the batch count through `safe_malloc()` and released through
`safe_free()`.

At plan initialization, the CPU computes the stage count and stores the $N-1$
stage-local twiddle factors. The FFT hot loop therefore performs no allocation
and calls neither `cosf()` nor `sinf()`. Linux/Spike selects
`csrc/nn_runtime_linux.c`; bare-metal selects
`baremetal/nn_runtime_baremetal.c`. Both provide 64-byte-aligned allocation and
the common `nn_runtime_read_cycles()` interface.

Regenerate the local input and ground truth with:

```sh
python3 py/fft/fft_groundtruth.py --size 1024 \
    --header testbench/fft_cpu/fft_vectors.h
```

The build script selects the `default` RV64GC CPU profile and passes
`NN_AUTO_VECTORIZE=OFF`. CMake consequently adds `-fno-tree-vectorize`,
`-fno-tree-slp-vectorize`, and `-fno-builtin`.

```sh
./scripts/configure_build.sh linux-pk cpu fft_cpu default
```

Run it with an RV64GC-compatible Spike and proxy kernel:

```sh
spike --isa=rv64gc_zicntr_zihpm pk \
    build/linux-pk/fft_cpu/default/cpu/static/fft_cpu
```

Build the pure scalar CPU bare-metal image with auto-vectorization disabled:

```sh
make baremetal \
    TESTBENCH=fft_cpu \
    BACKEND=cpu \
    HARDWARE_CONFIG=cpu
```

The generated images are:

```text
build/baremetal/fft_cpu/cpu_nn_cpu_baremetal.elf
build/baremetal/fft_cpu/cpu_nn_cpu_baremetal.bin
```

After identifying the intended unmounted whole SD-card device, flash the CPU
image with:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft_cpu \
    BACKEND=cpu \
    HARDWARE_CONFIG=cpu \
    SDCARD_DEVICE=/dev/sdc \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole SD-card device, not a partition. Flashing
overwrites data on the selected device.

The testbench reports the following regions separately:

- `twiddle plan cycles`: four runtime allocations and twiddle initialization;
- `input layout cycles`: copying 64 inputs into contiguous batch-major data;
- `FFT cycles`: 64 scalar FFT calls, including one bit reversal per input;
- `reused-plan total cycles`: input layout plus FFT;
- `first-run total cycles`: plan initialization, input layout, and FFT.

Result validation and printing are outside these measured regions. Validation
checks all 65,536 complex outputs and reports the batch and bin of mismatches.

## Performance results

### Current four-buffer runtime-plan implementation

> The measurements below were captured before bit reversal was fused into the
> input layout. Re-run the current binary before using these cycle counts.

```sh
spike --isa=rv64gc_zicntr_zihpm pk \
    build/linux-pk/fft_cpu/default/cpu/static/fft_cpu
```

| Environment | Configuration | FFT size | Batches | Twiddle plan cycles | Input layout cycles | FFT cycles | Reused-plan total cycles | First-run total cycles | FFT cycles/FFT | Reused-plan cycles/FFT | First-run cycles/FFT | Mismatches | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `rv64gc`, scalar FP32 | 1,024 | 64 | 114,235 | 839,015 | 15,432,206 | 16,271,221 | 16,385,456 | 241,128 | 254,237 | 256,022 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `HARDWARE_CONFIG=cpu`, scalar FP32, 50 MHz | 1,024 | 64 | 171,198 | 759,907 | 12,760,529 | 13,520,436 | 13,691,634 | 199,383 | 211,256 | 213,931 | 0 / 65,536 | PASS |

The maximum component error was `0.000024` at batch 0 bin 757, and the
maximum tolerance ratio was `0.385719` at batch 0 bin 1015. No non-finite
outputs were observed. Compared with the earlier 36,380,040-cycle scalar
kernel, precomputing twiddles reduces the FFT region by 57.58%, corresponding
to a 2.36x kernel speedup.

### Earlier dynamic-twiddle implementation

The following results predate the runtime-plan implementation and are retained
only as historical baselines. They recomputed twiddles inside the FFT hot loop;
new CPU measurements must be recorded before comparing against the current RVV
runtime-plan results.

| Environment | Configuration | Batches | FFT cycles | FFT cycles/FFT | Max component error | Max tolerance ratio | Mismatches | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `rv64gc`, scalar FP32 | 64 | 36,380,040 | 568,438 | 0.000024 | 0.385719 | 0 / 65,536 | PASS |

Each row is a single run rather than a multi-run average. All current CPU runs
reported zero non-finite outputs; the maximum component error was at batch 0
bin 757, and the maximum tolerance ratio was at batch 0 bin 1015.

`Max component error` and `Max tolerance ratio` can differ from the RVV values
because they depend on the exact operator sequence and FP32 rounding path. In
particular, scalar multiply/add expressions and RVV fused
multiply-add/subtract instructions need not produce identical last bits. The
correctness criterion is whether any point has a tolerance ratio greater than
1.0 or a non-finite result; zero mismatches means every complex output passed.

### Genesys2 FPGA configuration matrix

The following hardware configurations are planned for testing. A dash means
that no result has been recorded yet.

| Configuration | Batches | FFT cycles | FFT cycles/FFT | Max component error | Max tolerance ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 64 | — | — | — | — | Pending |
| `GENV128D128` | 64 | — | — | — | — | Pending |
| `GENV256D64` | 64 | — | — | — | — | Pending |
| `GENV256D128` | 64 | — | — | — | — | Pending |
| `GENV512D64` | 64 | — | — | — | — | Pending |
| `GENV512D128` | 64 | — | — | — | — | Pending |
| `LGVV128D128` | 64 | — | — | — | — | Pending |
| `LGVV256D128` | 64 | — | — | — | — | Pending |
| `LGVV512D128` | 64 | — | — | — | — | Pending |
