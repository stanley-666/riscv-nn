# Scalar CPU FFT testbench

This directory contains the scalar CPU implementation of the single-precision,
radix-2 FFT. `fft_cpu.c` contains no RVV intrinsics and reads its input and NumPy
ground truth from the local `fft_vectors.h`. This header is generated with the
same size and generator as the RVV FFT vectors, allowing direct comparison.

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

The reported cycle count surrounds only `fft_cpu_f32`; input copying, result
validation, and printing are excluded.

## Performance results

The result below uses the same 1024-point FP32 vectors as the RVV FFT. The run
passed with maximum component error `0.000024` at bin 757 and maximum tolerance
ratio `0.385719` at bin 1015.

| Environment | Configuration | Cycles | Result |
| --- | --- | ---: | --- |
| Spike simulation | `default` RV64GC scalar CPU | 576,027 | PASS |

This is a single reported run rather than an average over repeated trials.
