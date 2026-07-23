# Batched scaled int8 FFT

This testbench runs 64 independent 1024-point complex radix-2 FFTs with RVV
lanes spanning the batch dimension. Inputs, outputs, and twiddle factors use
signed Q7 (`int8_t`). Individual products widen to `int16_t` and complex
add/sub accumulation widens again to `int32_t`; every butterfly
stage divides its outputs by two, so an $N$-point transform has total scale
$1/N$ and cannot overflow merely because a butterfly adds two Q7 values.

For odd input $o=o_r+j o_i$ and Q7 twiddle $W=W_r+jW_i$, the rotated value is

$$
t_r=\mathrm{sat}_{8}\!\left(
  \mathrm{RNU}\frac{o_rW_r-o_iW_i}{2^7}\right),\qquad
t_i=\mathrm{sat}_{8}\!\left(
  \mathrm{RNU}\frac{o_iW_r+o_rW_i}{2^7}\right).
$$

The scaled butterfly is

$$
y_{\mathrm{upper}}=\mathrm{RNU}\frac{e+t}{2},\qquad
y_{\mathrm{lower}}=\mathrm{RNU}\frac{e-t}{2}.
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

The RVV binary constructs the Q7 twiddle table once, then runs baseline and
two-stage-fused kernels for each int32 accumulator LMUL:

- `m2`: `e8,mf2 -> e16,m1 -> e32,m2`;
- `m4`: `e8,m1 -> e16,m2 -> e32,m4`;
- `m8`: `e8,m2 -> e16,m4 -> e32,m8`.

Each of the six kernels runs 10 times and reports the integer average. Every
run reloads the original input and repeats bit reversal, while all runs share
the single twiddle plan. A fused kernel keeps the first stage's rounded,
saturated Q7 outputs in vector registers and feeds them directly to the second
stage. It removes only the intermediate memory store/reload: both stages still
perform their original RNU scaling and saturation, preserving bit-exact Q7
semantics. Bit-exact validation checks the final run outside the timed region.
The scalar `fft_cpu_int8` testbench provides matching baseline and fused
comparisons with compiler auto-vectorization disabled.

- `twiddle plan cycles`: allocation and dynamic Q7 twiddle generation.
- `input layout cycles`: copy into contiguous `[bin][batch]` rows.
- `bit reversal cycles`: only the bit-reversal row-swap phase.
- `butterfly cycles`: twiddle lookup and all scaled butterfly stages.
- `FFT cycles`: bit reversal plus butterfly cycles.
- the current six-variant binary reports average input-layout,
  bit-reversal, butterfly, and FFT cycles separately.

Performance rows should only be added after bit-exact validation reports
`0 / 65536` mismatched complex points.

## Performance results

### Current six-variant Spike result

All variants below passed bit-exact validation with `0 / 65,536` mismatches.
The values are 10-run averages; twiddle-plan creation took 110,925 cycles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Fusion effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 671,716 | 106,872 | 1,637,494 | 1,744,366 | baseline |
| m2 accumulator stage-fused | 665,608 | 106,872 | 1,345,290 | 1,452,162 | 1.22x faster |
| m4 accumulator | 665,608 | 106,872 | 846,898 | 953,770 | baseline |
| m4 accumulator stage-fused | 665,608 | 106,872 | 707,266 | 814,138 | 1.20x faster |
| m8 accumulator | 665,608 | 106,872 | **481,308** | **588,180** | baseline, fastest |
| m8 accumulator stage-fused | 665,608 | 106,872 | 512,184 | 619,056 | 1.06x slower |

The result follows the FP32 trend: fusion helps m2 and m4, but m8 register
pressure outweighs the removed stage-boundary traffic. Baseline m8 remains the
fastest int8 RVV kernel on Spike.

### Genesys2 nine-configuration fusion campaign

All rows use a 50 MHz clock, 1,024-point scaled Q7 FFTs, 64 batches, one shared
twiddle plan, and 10-run averages. A hardware configuration is complete only
when all six variants pass bit-exact validation with zero mismatches.

| Hardware config | VLEN | Datapath | Complete six-variant result |
| --- | ---: | ---: | --- |
| `GENV128D64` | 128 | 64 | PASS, 0 / 65,536 for all variants |
| `GENV128D128` | 128 | 128 | PASS, 0 / 65,536 for all variants |
| `GENV256D64` | 256 | 64 | PASS, 0 / 65,536 for all variants |
| `GENV256D128` | 256 | 128 | PASS, 0 / 65,536 for all variants |
| `GENV512D64` | 512 | 64 | PASS, 0 / 65,536 for all variants |
| `GENV512D128` | 512 | 128 | PASS, 0 / 65,536 for all variants |
| `LGVV128D128` | 128 | 128 | PASS, 0 / 65,536 for all variants |
| `LGVV256D128` | 256 | 128 | PASS, 0 / 65,536 for all variants |
| `LGVV512D128` | 512 | 128 | PASS, 0 / 65,536 for all variants |

#### Main int8 FFT butterfly table

Values are average butterfly cycles for the complete 10-stage transform over
64 FFTs. Input layout, bit reversal, validation, and one-time twiddle-plan
creation are excluded. Lower is better.

| Hardware config | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 3,218,163 | 2,736,857 | 2,450,801 | 2,393,286 | **2,243,324** | 2,416,877 | m8 |
| `GENV128D128` | 2,480,194 | 1,784,307 | 1,721,265 | 1,458,510 | **1,367,989** | 1,495,739 | m8 |
| `GENV256D64` | 2,454,832 | 2,383,223 | **2,113,540** | 2,212,496 | 2,193,577 | 2,262,606 | m4 |
| `GENV256D128` | 1,655,918 | 1,394,549 | 1,262,530 | 1,220,177 | **1,116,785** | 1,236,382 | m8 |
| `GENV512D64` | 2,112,668 | 2,207,872 | **2,049,368** | 2,113,627 | 2,324,605 | 2,449,141 | m4 |
| `GENV512D128` | 1,266,876 | 1,215,769 | **1,054,211** | 1,121,609 | 1,227,415 | 1,331,715 | m4 |
| `LGVV128D128` | 2,050,961 | 1,784,323 | 1,365,847 | 1,458,513 | **1,322,047** | 1,495,291 | m8 |
| `LGVV256D128` | 1,310,052 | 1,394,540 | **1,116,502** | 1,220,206 | 1,116,715 | 1,236,999 | m4 |
| `LGVV512D128` | 1,124,729 | 1,215,757 | **1,043,142** | 1,121,857 | 1,227,334 | 1,331,394 | m4 |

#### Complete timing data for the other eight configurations

Every row below is a cold-boot Genesys2 measurement averaged over 10 runs and
passed bit-exact validation with `0 / 65,536` mismatches.

| Config | Variant | Input layout | Bit reversal | Butterfly | FFT |
| --- | --- | ---: | ---: | ---: | ---: |
| `GENV128D64` | m2 | 1,143,422 | 120,092 | 3,218,163 | 3,338,255 |
| `GENV128D64` | m2 fused | 1,142,582 | 119,749 | 2,736,857 | 2,856,606 |
| `GENV128D64` | m4 | 1,142,687 | 119,665 | 2,450,801 | 2,570,466 |
| `GENV128D64` | m4 fused | 1,142,456 | 119,707 | 2,393,286 | 2,512,993 |
| `GENV128D64` | m8 | 1,142,491 | 119,635 | **2,243,324** | **2,362,959** |
| `GENV128D64` | m8 fused | 1,142,516 | 119,669 | 2,416,877 | 2,536,546 |
| `GENV128D128` | m2 | 1,141,790 | 111,400 | 2,480,194 | 2,591,594 |
| `GENV128D128` | m2 fused | 1,140,495 | 111,090 | 1,784,307 | 1,895,397 |
| `GENV128D128` | m4 | 1,140,594 | 111,056 | 1,721,265 | 1,832,321 |
| `GENV128D128` | m4 fused | 1,140,597 | 111,083 | 1,458,510 | 1,569,593 |
| `GENV128D128` | m8 | 1,140,474 | 111,131 | **1,367,989** | **1,479,120** |
| `GENV128D128` | m8 fused | 1,140,566 | 111,129 | 1,495,739 | 1,606,868 |
| `GENV256D64` | m2 | 1,143,394 | 100,018 | 2,454,832 | 2,554,850 |
| `GENV256D64` | m2 fused | 1,142,659 | 99,621 | 2,383,223 | 2,482,844 |
| `GENV256D64` | m4 | 1,142,589 | 99,663 | **2,113,540** | **2,213,203** |
| `GENV256D64` | m4 fused | 1,142,573 | 99,588 | 2,212,496 | 2,312,084 |
| `GENV256D64` | m8 | 1,142,491 | 99,555 | 2,193,577 | 2,293,132 |
| `GENV256D64` | m8 fused | 1,142,493 | 99,523 | 2,262,606 | 2,362,129 |
| `GENV256D128` | m2 | 1,141,913 | 91,575 | 1,655,918 | 1,747,493 |
| `GENV256D128` | m2 fused | 1,140,451 | 91,300 | 1,394,549 | 1,485,849 |
| `GENV256D128` | m4 | 1,140,495 | 91,264 | 1,262,530 | 1,353,794 |
| `GENV256D128` | m4 fused | 1,140,495 | 91,196 | 1,220,177 | 1,311,373 |
| `GENV256D128` | m8 | 1,140,463 | 91,291 | **1,116,785** | **1,208,076** |
| `GENV256D128` | m8 fused | 1,140,524 | 91,287 | 1,236,382 | 1,327,669 |
| `GENV512D64` | m2 | 1,143,516 | 65,271 | 2,112,668 | 2,177,939 |
| `GENV512D64` | m2 fused | 1,142,805 | 65,041 | 2,207,872 | 2,272,913 |
| `GENV512D64` | m4 | 1,142,614 | 65,050 | **2,049,368** | **2,114,418** |
| `GENV512D64` | m4 fused | 1,142,605 | 65,085 | 2,113,627 | 2,178,712 |
| `GENV512D64` | m8 | 1,142,621 | 65,046 | 2,324,605 | 2,389,651 |
| `GENV512D64` | m8 fused | 1,142,574 | 65,086 | 2,449,141 | 2,514,227 |
| `GENV512D128` | m2 | 1,141,786 | 63,712 | 1,266,876 | 1,330,588 |
| `GENV512D128` | m2 fused | 1,140,754 | 63,608 | 1,215,769 | 1,279,377 |
| `GENV512D128` | m4 | 1,140,381 | 63,653 | **1,054,211** | **1,117,864** |
| `GENV512D128` | m4 fused | 1,140,530 | 63,671 | 1,121,609 | 1,185,280 |
| `GENV512D128` | m8 | 1,140,532 | 63,645 | 1,227,415 | 1,291,060 |
| `GENV512D128` | m8 fused | 1,140,340 | 63,689 | 1,331,715 | 1,395,404 |
| `LGVV128D128` | m2 | 1,141,910 | 111,450 | 2,050,961 | 2,162,411 |
| `LGVV128D128` | m2 fused | 1,140,604 | 111,087 | 1,784,323 | 1,895,410 |
| `LGVV128D128` | m4 | 1,140,623 | 111,060 | 1,365,847 | 1,476,907 |
| `LGVV128D128` | m4 fused | 1,140,511 | 111,114 | 1,458,513 | 1,569,627 |
| `LGVV128D128` | m8 | 1,140,424 | 111,072 | **1,322,047** | **1,433,119** |
| `LGVV128D128` | m8 fused | 1,140,533 | 111,145 | 1,495,291 | 1,606,436 |
| `LGVV256D128` | m2 | 1,141,683 | 91,571 | 1,310,052 | 1,401,623 |
| `LGVV256D128` | m2 fused | 1,140,713 | 91,264 | 1,394,540 | 1,485,804 |
| `LGVV256D128` | m4 | 1,140,638 | 91,209 | **1,116,502** | **1,207,711** |
| `LGVV256D128` | m4 fused | 1,140,610 | 91,274 | 1,220,206 | 1,311,480 |
| `LGVV256D128` | m8 | 1,140,522 | 91,268 | 1,116,715 | 1,207,983 |
| `LGVV256D128` | m8 fused | 1,140,442 | 91,281 | 1,236,999 | 1,328,280 |

Twiddle-plan cycles were 186,418 (`GENV128D64`), 186,294
(`GENV128D128`), 186,430 (`GENV256D64`), 186,394 (`GENV256D128`),
186,464 (`GENV512D64`), 186,294 (`GENV512D128`), 186,313
(`LGVV128D128`), and 186,561 (`LGVV256D128`).

#### Complete LGVV512D128 timing data

This Genesys2 FPGA measurement uses VLEN=512, D=128, cold boot, and one shared
twiddle plan. Twiddle-plan creation took 186,391 cycles. All six variants
passed bit-exact validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 1,141,766 | 63,710 | 1,124,729 | 1,188,439 | 0 / 65,536 |
| m2 accumulator stage-fused | 1,140,433 | 63,618 | 1,215,757 | 1,279,375 | 0 / 65,536 |
| m4 accumulator | 1,140,523 | 63,589 | **1,043,142** | **1,106,731** | 0 / 65,536 |
| m4 accumulator stage-fused | 1,140,439 | 63,638 | 1,121,857 | 1,185,495 | 0 / 65,536 |
| m8 accumulator | 1,140,606 | 63,586 | 1,227,334 | 1,290,920 | 0 / 65,536 |
| m8 accumulator stage-fused | 1,140,470 | 63,652 | 1,331,394 | 1,395,046 | 0 / 65,536 |

On LGVV512D128, fusion makes m2 `1.08x`, m4 `1.08x`, and m8 `1.08x` slower.
Baseline m4 is the fastest kernel: it is `1.08x` faster than baseline m2 and
`1.18x` faster than baseline m8. Unlike Spike, where fusion helps m2 and m4,
this FPGA result shows that the extra register pressure and instruction
schedule outweigh the eliminated stage-boundary traffic for every LMUL.

#### Nine-configuration observations

All nine configurations and all 54 measured variants passed bit-exact
validation. A fused kernel is never the overall winner for a hardware
configuration, even though fusion improves selected m2 or m4 baselines. The
best choices are:

- baseline m8 for `GENV128D64`, `GENV128D128`, `GENV256D128`, and
  `LGVV128D128`;
- baseline m4 for `GENV256D64`, `GENV512D64`, `GENV512D128`,
  `LGVV256D128`, and `LGVV512D128`.

The fastest hardware result is LGVV512D128 baseline m4 at 1,043,142
butterfly cycles. GENV512D128 baseline m4 is within 1.1% at 1,054,211 cycles.
At VLEN=256, LGVV m4 and m8 differ by only 213 cycles, showing a throughput
plateau rather than proportional scaling with LMUL. These results support
choosing LMUL from the complete dataflow and register pressure, not simply
selecting the largest accumulator LMUL.

Using the current 50 MHz scalar CPU result, baseline RVV provides a `10.99x`
butterfly speedup (`11,461,004 / 1,043,142`) and a `12.47x` complete-FFT
speedup (`13,801,835 / 1,106,731`). When both sides use stage fusion, the best
RVV fused kernel is GENV512D128 m4 fused: it provides a `10.38x` butterfly
speedup (`11,638,084 / 1,121,609`) and an `11.76x` complete-FFT speedup
(`13,940,562 / 1,185,280`). Baseline-to-baseline is the primary hardware
comparison because fusion is slower for both the scalar CPU and the globally
fastest RVV configuration.

### Archived pre-fusion results

The following rows predate the six-variant fusion implementation:

> These rows predate the hot-loop branch hoist and separate bit-reversal /
> butterfly counters. Re-run the current binary before using them for the
> final CPU/RVV or FP32/int8 comparison.

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
