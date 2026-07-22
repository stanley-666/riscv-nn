# Scalar CPU FFT testbench

This directory contains a scalar CPU batch of 64 independent, single-precision
radix-2 FFTs. `fft_cpu.c` contains no RVV intrinsics and reads its input and
NumPy ground truth from the local `fft_vectors.h`. This header is generated with
the same size and generator as the RVV FFT vectors, allowing direct comparison.

The CPU keeps its natural `[batch][bin]` layout and executes one FFT per batch;
it does not perform the RVV implementation's `[batch][bin]` to `[bin][batch]`
transpose. Input replication is outside the measured region. Reported FFT
cycles therefore cover exactly 64 scalar FFT calls and include a per-FFT value.

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

The reported cycle count surrounds only the loop of 64 `fft_cpu_f32` calls;
input copying, result validation, and printing are excluded. Validation checks
all 65,536 complex outputs and reports the batch and bin of mismatches.

## Performance results

The previous single-FFT result is not directly comparable to the current
64-batch testbench and has been removed. The current scalar batch result is:

| Environment | Configuration | Batches | FFT cycles | FFT cycles/FFT | Max component error | Max tolerance ratio | Mismatches | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `rv64gc`, scalar FP32 | 64 | 36,380,040 | 568,438 | 0.000024 | 0.385719 | 0 / 65,536 | PASS |
| Genesys2 FPGA | `HARDWARE_CONFIG=cpu`, scalar FP32, 50 MHz, cold cache | 64 | 60,753,733 | 949,277 | 0.000024 | 0.385719 | 0 / 65,536 | PASS |

Each row is a single run rather than a multi-run average. Both reported zero
non-finite outputs; the maximum component error was at batch 0 bin 757, and the
maximum tolerance ratio was at batch 0 bin 1015.

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
