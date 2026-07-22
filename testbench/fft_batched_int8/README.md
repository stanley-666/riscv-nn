# Batched scaled int8 FFT

This testbench runs 64 independent 1024-point complex radix-2 FFTs with RVV
lanes spanning the batch dimension. Inputs, outputs, and twiddle factors use
signed Q7 (`int8_t`). Individual products widen to `int16_t` and complex
add/sub accumulation widens again to `int32_t`; every butterfly
stage divides its outputs by two, so an $N$-point transform has total scale
$1/N$ and cannot overflow merely because a butterfly adds two Q7 values.

For odd input $o=o_r+j o_i$ and Q7 twiddle $W=W_r+jW_i$, the rotated value is

$$
t_r=\operatorname{sat}_{8}\!\left(
  \operatorname{RNU}\frac{o_rW_r-o_iW_i}{2^7}\right),\qquad
t_i=\operatorname{sat}_{8}\!\left(
  \operatorname{RNU}\frac{o_iW_r+o_rW_i}{2^7}\right).
$$

The scaled butterfly is

$$
y_{\mathrm{upper}}=\operatorname{RNU}\frac{e+t}{2},\qquad
y_{\mathrm{lower}}=\operatorname{RNU}\frac{e-t}{2}.
$$

`RNU` is the RVV round-to-nearest-up mode. The unity-twiddle path bypasses the
Q7 representation of $+1$ and directly applies the scaled add/subtract. The
implementation uses `vaadd`/`vasub` for scaled butterflies, `vwmul` plus
32-bit sign extension for overflow-safe Q7 products, and `vnclip` for rounded
saturating conversion back to int8.

## Generate bit-exact vectors

```sh
python3 py/fft/fft_int8_groundtruth.py \
    --size 1024 \
    --header testbench/fft_batched_int8/fft_int8_vectors.h
```

The generated header contains 64-byte-aligned real/imaginary input and
bit-exact ground-truth arrays. The Python model applies the same bit reversal,
Q7 twiddle quantization, RNU rounding, saturation, and per-stage scaling as the
RVV kernel. Validation requires exact equality and prints the first 64
mismatched complex points with batch, bin, actual, and expected values.

## Linux/pk and Spike

```sh
./scripts/configure_build.sh linux-pk vector fft_batched_int8 zvl128b

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_batched_int8/zvl128b/vector/static/fft_batched_int8
```

## Bare-metal build

```sh
./scripts/configure_build.sh baremetal vector fft_batched_int8 V128D128B
```

The binary is written to:

```text
build/baremetal/fft_batched_int8/V128D128B_nn_rvv_baremetal.bin
```

## Bare-metal flash

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft_batched_int8 \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. Flashing
overwrites data on the selected device.

## Measured regions

- `twiddle plan cycles`: allocation and dynamic Q7 twiddle generation.
- `input layout cycles`: copy into contiguous `[bin][batch]` rows.
- `FFT cycles`: bit-reversal row swaps and all scaled butterfly stages.
- `reused-plan total cycles`: input layout plus FFT.
- `first-run total cycles`: plan creation plus reused-plan total.

Performance rows should only be added after bit-exact validation reports
`0 / 65536` mismatched complex points.

## Performance results

The following is a single Spike run of the current implementation:

| Environment | Configuration | FFT size | Batches | Scale | Twiddle plan cycles | Input layout cycles | FFT cycles | Reused-plan total cycles | First-run total cycles | FFT cycles/FFT | Reused-plan cycles/FFT | First-run cycles/FFT | Mismatches | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 113,960 | 400,034 | 545,484 | 945,518 | 1,059,478 | 8,523 | 14,773 | 16,554 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV128D64`, VLEN=128, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 185,848 | 387,932 | 2,324,473 | 2,712,405 | 2,898,253 | 36,319 | 42,381 | 45,285 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV128D128`, VLEN=128, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 185,901 | 390,813 | 1,423,981 | 1,814,794 | 2,000,695 | 22,249 | 28,356 | 31,260 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV256D64`, VLEN=256, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 185,836 | 388,051 | 2,264,158 | 2,652,209 | 2,838,045 | 35,377 | 41,440 | 44,344 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV256D128`, VLEN=256, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 186,003 | 391,197 | 1,182,192 | 1,573,389 | 1,759,392 | 18,471 | 24,584 | 27,490 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV512D64`, VLEN=512, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 186,041 | 388,579 | 2,384,805 | 2,773,384 | 2,959,425 | 37,262 | 43,334 | 46,241 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV512D128`, VLEN=512, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 186,048 | 390,578 | 1,283,396 | 1,673,974 | 1,860,022 | 20,053 | 26,155 | 29,062 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV128D128`, VLEN=128, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 186,000 | 390,874 | 1,403,835 | 1,794,709 | 1,980,709 | 21,934 | 28,042 | 30,948 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV256D128Shuttle`, VLEN=256, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 185,844 | 391,088 | 1,181,781 | 1,572,869 | 1,758,713 | 18,465 | 24,576 | 27,479 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV512D128`, VLEN=512, Q7 data m2 / int32 accumulator m8 | 1,024 | 64 | $1/1024$ | 186,018 | 390,713 | 1,282,125 | 1,672,838 | 1,858,856 | 20,033 | 26,138 | 29,044 | 0 / 65,536 | PASS |

The maximum component error is zero because validation uses the same bit-exact
Q7 arithmetic definition as the Python ground truth.

For context, the current FP32 `zvl128b` run uses 517,852 FFT cycles, 1,094,255
reused-plan cycles, and 1,200,155 first-run cycles. The scaled int8 kernel is
5.34% slower in this Spike run because its overflow-safe complex multiply
widens through int16 into int32 and performs two narrowing steps. However, its
smaller input layout reduces the complete reused-plan path by 13.59% and the
first-run path by 11.72%.
