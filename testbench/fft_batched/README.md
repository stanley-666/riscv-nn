# Batched RVV FFT testbench

This testbench runs 64 independent 1024-point FP32 radix-2 FFTs. Every batch
uses the same input and NumPy ground truth as `testbench/fft`.

The original batch-major input `[batch][bin]` is transposed in a measured
preprocessing step into `[bin][batch]`. RVV operations then span the batch
dimension, providing parallel work even during the early FFT stages.

## Build and run

```sh
./scripts/configure_build.sh linux-pk vector fft_batched zvl128b

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_batched/zvl128b/vector/static/fft_batched
```

## Performance results

The result below is a single Spike run with 64 identical 1024-point FP32 FFTs.
All batches passed with maximum component error `0.000024` at batch 0 bin 757
and maximum tolerance ratio `0.385719` at batch 0 bin 1015.

| Environment | Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, LMUL=m4 | 64 | 827,078 | 1,104,678 | 1,931,756 | 17,260 | 30,183 | PASS |

Input initialization, validation, and bin printing are outside the measured
regions. These values are from one run rather than an average.
