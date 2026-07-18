# RVV DFT GEMV testbench

This testbench expresses the 1024-point complex DFT as Conv1D-style
scalar-vector MAC operations. The original input arrays remain unchanged. Each
scalar input sample multiplies contiguous vectors of twiddle weights and
accumulates output frequency bins with `vfmacc.vf` and `vfnmsac.vf`.

The natural twiddle layout `[output_bin][input_index]` is transposed in a
measured preprocessing step into `[input_index][output_bin]`, matching the
`(input channel, output channel)` weight layout used by the RVV Conv1D kernel.

This produces the same DFT values as the radix-2 FFT, but performs `O(N^2)`
work rather than the FFT's `O(N log N)` work.

Because each GEMV output accumulates 1024 FP32 terms in a different order from
the radix-2 FFT, results are mathematically equivalent but not bit-exact. This
test uses `atol=2e-4` and `rtol=1e-6`; batch printing retains the same
`bin N: real +/-imagj` format as the FFT testbench.

```sh
python3 py/fft/fft_groundtruth.py --size 1024 \
    --header testbench/fft_gemv/fft_vectors.h
```
```sh
./scripts/configure_build.sh linux-pk vector fft_gemv zvl128b
```
```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_gemv/zvl128b/vector/static/fft_gemv
```

The output separately reports twiddle-generation, memory-reorder, GEMV-kernel,
and end-to-end cycles. Validation and bin printing are outside all measurements.

## Performance results

The result below is a single Spike run of the 1024-point FP32 complex DFT GEMV.
It passed with maximum component error `0.000145` at bin 297 and maximum
tolerance ratio `0.621375` at bin 297.

| Environment | Configuration | Twiddle generation cycles | Memory reorder cycles | GEMV cycles | Total cycles | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, LMUL=m8 | 252,806,463 | 13,262,858 | 438,558 | 266,507,879 | PASS |

The total includes runtime construction of the full 1024x1024 complex twiddle
matrix. Validation and bin printing are excluded. These values are from one run
rather than an average.
