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

### Dense mixed-radix Spike ablation

The RVV binary now includes the same dense mixed-radix 4/16/16 transform used
by the CPU and Gemmini experiments. It vectorizes the 64 batches while
performing each dense radix transform with widening INT32 accumulators.

The following Spike run uses VLEN=512:

| RVV Spike variant | FFT cycles | Relative to mixed radix |
| --- | ---: | ---: |
| Radix-2 m4 baseline | 386,632 | 4.44x faster |
| Radix-2 m4 stage-fused | **322,759** | **5.31x faster** |
| Dense mixed radix 4/16/16 m8 | 1,715,067 | reference |

The mixed-radix RVV output has 0 / 65,536 mismatches against the shared scalar
mixed-radix reference. Its precision difference from the radix-2 output is
the same as the CPU and Gemmini paths: 60,992 / 65,536 points are within +/-1
per component, mean absolute component difference is 0.533, and maximum
component difference is 3.

Unlike Gemmini, RVV does not need a dense transform to expose batch
parallelism. Radix-2 already maps each twiddle scalar across the 64-batch
vector and avoids zero MACs. The direct dense radix-16 arithmetic therefore
increases work and register pressure, making mixed radix slower on RVV.

### Batch-major layout Spike ablation

An additional radix-2 kernel stores data as `[batch][bin]` and vectorizes
butterfly offsets within each FFT. It uses vector-vector twiddle
multiplication, whereas the primary `[bin][batch]` kernel broadcasts one
twiddle scalar across the 64-batch vector.

| LMUL | Bin-major FFT cycles | Batch-major FFT cycles | Batch-major penalty |
| --- | ---: | ---: | ---: |
| m2 | 621,666 | 5,080,368 | 8.17x |
| m4 | **386,632** | 4,963,632 | 12.84x |
| m8 | 407,111 | **4,919,856** | 12.08x |

All VLEN=512 Spike variants passed bit-exact Q7 validation. Comparing the best
kernel from each layout, batch-major is 12.72x slower. Its butterfly region is
13.21x slower, while its preindexed out-of-place bit reversal is 11.24x slower
than bin-major's vector row swap. A corresponding LGVV512D128 FPGA layout
measurement is reported below.

### Genesys2 nine-configuration fusion campaign

All rows use a 50 MHz clock, 1,024-point scaled Q7 FFTs, 64 batches, shared
runtime plans, and 10-run averages. A current hardware configuration is
complete only when all ten layout/radix variants pass their corresponding
validation.

| Hardware config | VLEN | Datapath | Complete current result |
| --- | ---: | ---: | --- |
| `GENV128D64` | 128 | 64 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `GENV128D128` | 128 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `GENV256D64` | 256 | 64 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `GENV256D128` | 256 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `GENV512D64` | 512 | 64 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `GENV512D128` | 512 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `LGVV128D128` | 128 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `LGVV256D128` | 256 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |
| `LGVV512D128` | 512 | 128 | PASS, 0 / 65,536 for all 10 layout/radix variants |

#### Main int8 FFT butterfly table

Values are average butterfly cycles for the complete 10-stage transform over
64 FFTs. Input layout, bit reversal, validation, and one-time twiddle-plan
creation are excluded. Lower is better.

| Hardware config | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 2,698,355 | 2,665,702 | 2,261,688 | 2,361,732 | **2,230,022** | 2,525,973 | m8 |
| `GENV128D128` | 1,954,008 | 1,720,619 | 1,463,171 | 1,412,149 | **1,334,925** | 1,605,074 | m8 |
| `GENV256D64` | 2,266,102 | 2,355,220 | **2,083,860** | 2,203,580 | 2,191,758 | 2,379,189 | m4 |
| `GENV256D128` | 1,395,736 | 1,365,359 | 1,162,029 | 1,210,619 | **1,114,775** | 1,310,252 | m8 |
| `GENV512D64` | 2,083,445 | 2,206,151 | **2,049,658** | 2,118,149 | 2,322,690 | 2,623,901 | m4 |
| `GENV512D128` | 1,159,181 | 1,210,407 | **1,041,871** | 1,124,884 | 1,225,320 | 1,462,279 | m4 |
| `LGVV128D128` | 1,589,855 | 1,720,779 | **1,294,087** | 1,411,793 | 1,318,809 | 1,605,037 | m4 |
| `LGVV256D128` | 1,229,055 | 1,365,249 | **1,089,147** | 1,208,274 | 1,114,749 | 1,310,245 | m4 |
| `LGVV512D128` | 1,099,285 | 1,208,405 | **1,036,779** | 1,125,477 | 1,225,322 | 1,446,335 | m4 |

#### Complete timing data for the other eight configurations

Every row below is a cold-boot Genesys2 measurement averaged over 10 runs and
passed bit-exact validation with `0 / 65,536` mismatches.

| Config | Variant | Input layout | Bit reversal | Butterfly | FFT |
| --- | --- | ---: | ---: | ---: | ---: |
| `GENV128D64` | m2 | 364,615 | 89,509 | 2,698,355 | 2,787,864 |
| `GENV128D64` | m2 fused | 359,084 | 89,432 | 2,665,702 | 2,755,134 |
| `GENV128D64` | m4 | 356,431 | 89,648 | 2,261,688 | 2,351,336 |
| `GENV128D64` | m4 fused | 356,349 | 89,081 | 2,361,732 | 2,450,813 |
| `GENV128D64` | m8 | 356,377 | 90,873 | **2,230,022** | **2,320,895** |
| `GENV128D64` | m8 fused | 361,790 | 89,717 | 2,525,973 | 2,615,690 |
| `GENV128D128` | m2 | 364,759 | 81,445 | 1,954,008 | 2,035,453 |
| `GENV128D128` | m2 fused | 358,634 | 81,365 | 1,720,619 | 1,801,984 |
| `GENV128D128` | m4 | 356,140 | 80,577 | 1,463,171 | 1,543,748 |
| `GENV128D128` | m4 fused | 356,225 | 80,786 | 1,412,149 | 1,492,935 |
| `GENV128D128` | m8 | 356,216 | 83,682 | **1,334,925** | **1,418,607** |
| `GENV128D128` | m8 fused | 361,598 | 80,905 | 1,605,074 | 1,685,979 |
| `GENV256D64` | m2 | 364,658 | 67,751 | 2,266,102 | 2,333,853 |
| `GENV256D64` | m2 fused | 359,438 | 69,517 | 2,355,220 | 2,424,737 |
| `GENV256D64` | m4 | 356,370 | 67,547 | **2,083,860** | **2,151,407** |
| `GENV256D64` | m4 fused | 356,392 | 69,263 | 2,203,580 | 2,272,843 |
| `GENV256D64` | m8 | 356,226 | 70,397 | 2,191,758 | 2,262,155 |
| `GENV256D64` | m8 fused | 361,865 | 68,927 | 2,379,189 | 2,448,116 |
| `GENV256D128` | m2 | 364,744 | 61,341 | 1,395,736 | 1,457,077 |
| `GENV256D128` | m2 fused | 358,801 | 61,031 | 1,365,359 | 1,426,390 |
| `GENV256D128` | m4 | 356,420 | 61,314 | 1,162,029 | 1,223,343 |
| `GENV256D128` | m4 fused | 356,184 | 61,079 | 1,210,619 | 1,271,698 |
| `GENV256D128` | m8 | 356,264 | 63,505 | **1,114,775** | **1,178,280** |
| `GENV256D128` | m8 fused | 361,596 | 61,520 | 1,310,252 | 1,371,772 |
| `GENV512D64` | m2 | 364,595 | 47,524 | 2,083,445 | 2,130,969 |
| `GENV512D64` | m2 fused | 358,862 | 47,396 | 2,206,151 | 2,253,547 |
| `GENV512D64` | m4 | 356,269 | 47,417 | **2,049,658** | **2,097,075** |
| `GENV512D64` | m4 fused | 356,303 | 47,459 | 2,118,149 | 2,165,608 |
| `GENV512D64` | m8 | 356,303 | 49,280 | 2,322,690 | 2,371,970 |
| `GENV512D64` | m8 fused | 361,909 | 47,395 | 2,623,901 | 2,671,296 |
| `GENV512D128` | m2 | 364,682 | 37,646 | 1,159,181 | 1,196,827 |
| `GENV512D128` | m2 fused | 359,318 | 37,808 | 1,210,407 | 1,248,215 |
| `GENV512D128` | m4 | 356,244 | 37,648 | **1,041,871** | **1,079,519** |
| `GENV512D128` | m4 fused | 356,212 | 37,625 | 1,124,884 | 1,162,509 |
| `GENV512D128` | m8 | 356,178 | 39,652 | 1,225,320 | 1,264,972 |
| `GENV512D128` | m8 fused | 361,535 | 37,758 | 1,462,279 | 1,500,037 |
| `LGVV128D128` | m2 | 364,707 | 81,499 | 1,589,855 | 1,671,354 |
| `LGVV128D128` | m2 fused | 359,450 | 81,410 | 1,720,779 | 1,802,189 |
| `LGVV128D128` | m4 | 356,142 | 80,493 | **1,294,087** | **1,374,580** |
| `LGVV128D128` | m4 fused | 356,121 | 80,846 | 1,411,793 | 1,492,639 |
| `LGVV128D128` | m8 | 356,157 | 83,642 | 1,318,809 | 1,402,451 |
| `LGVV128D128` | m8 fused | 361,500 | 80,907 | 1,605,037 | 1,685,944 |
| `LGVV256D128` | m2 | 364,771 | 61,288 | 1,229,055 | 1,290,343 |
| `LGVV256D128` | m2 fused | 359,046 | 61,104 | 1,365,249 | 1,426,353 |
| `LGVV256D128` | m4 | 356,162 | 61,375 | **1,089,147** | **1,150,522** |
| `LGVV256D128` | m4 fused | 356,225 | 61,100 | 1,208,274 | 1,269,374 |
| `LGVV256D128` | m8 | 356,197 | 63,498 | 1,114,749 | 1,178,247 |
| `LGVV256D128` | m8 fused | 361,657 | 61,510 | 1,310,245 | 1,371,755 |

The current extended binaries report 5,122,025 cycles for GENV128D64,
5,124,963 cycles for GENV128D128, 5,122,211 cycles for GENV256D64,
5,122,377 cycles for GENV256D128, 5,122,413 cycles for GENV512D64,
5,124,717 cycles for GENV512D128, 5,125,874 cycles for LGVV128D128, and
5,122,400 cycles for LGVV256D128; each is the combined radix-2, mixed-radix,
and reverse-index initialization region rather than a twiddle-only plan.

#### Complete GENV128D64 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=128, D=64, cold boot, and 10-run
averages. The combined initialization region took 5,122,025 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,615 | 89,509 | 2,698,355 | 2,787,864 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,084 | 89,432 | 2,665,702 | 2,755,134 | 0 / 65,536 |
| m4 accumulator | 356,431 | 89,648 | 2,261,688 | 2,351,336 | 0 / 65,536 |
| m4 accumulator stage-fused | **356,349** | **89,081** | 2,361,732 | 2,450,813 | 0 / 65,536 |
| m8 accumulator | 356,377 | 90,873 | **2,230,022** | **2,320,895** | 0 / 65,536 |
| m8 accumulator stage-fused | 361,790 | 89,717 | 2,525,973 | 2,615,690 | 0 / 65,536 |

Fusion improves m2 by `1.01x`, but makes m4 `1.04x` and m8 `1.13x` slower.
Baseline m8 is the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 544,212 | 1,232,098 | 8,695,918 | 9,928,016 | 0 / 65,536 |
| m4 accumulator | **544,148** | 1,231,844 | **8,118,857** | **9,350,701** | 0 / 65,536 |
| m8 accumulator | 544,483 | **1,231,348** | 10,502,905 | 11,734,253 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `4.03x` slower in complete FFT cycles and `3.64x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`3.56x` slower for m2, `3.98x` slower for m4, and `5.06x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,138,642 | 15,104,587 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `6.51x` slower than the best bin-major radix-2
kernel and `1.62x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

At the same VLEN=128, GENV128D64 requires `1.67x` as many best bin-major
butterfly cycles and `1.64x` as many complete FFT cycles as GENV128D128.

#### Complete GENV128D128 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=128, D=128, cold boot, and 10-run
averages. The combined initialization region took 5,124,963 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,759 | 81,445 | 1,954,008 | 2,035,453 | 0 / 65,536 |
| m2 accumulator stage-fused | 358,634 | 81,365 | 1,720,619 | 1,801,984 | 0 / 65,536 |
| m4 accumulator | **356,140** | **80,577** | 1,463,171 | 1,543,748 | 0 / 65,536 |
| m4 accumulator stage-fused | 356,225 | 80,786 | 1,412,149 | 1,492,935 | 0 / 65,536 |
| m8 accumulator | 356,216 | 83,682 | **1,334,925** | **1,418,607** | 0 / 65,536 |
| m8 accumulator stage-fused | 361,598 | 80,905 | 1,605,074 | 1,685,979 | 0 / 65,536 |

Fusion improves m2 by `1.14x` and m4 by `1.04x`, while fused m8 is `1.20x`
slower. Baseline m8 is the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 544,288 | 1,231,930 | 7,730,108 | 8,962,038 | 0 / 65,536 |
| m4 accumulator | **543,832** | 1,232,281 | **7,253,091** | **8,485,372** | 0 / 65,536 |
| m8 accumulator | 544,337 | **1,231,509** | 8,811,832 | 10,043,341 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `5.98x` slower in complete FFT cycles and `5.43x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`4.40x` slower for m2, `5.50x` slower for m4, and `7.08x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,817 | 8,342,603 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `5.88x` slower than the best bin-major radix-2
kernel, but `1.02x` faster than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Complete GENV256D64 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=256, D=64, cold boot, and 10-run
averages. The combined initialization region took 5,122,211 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,658 | 67,751 | 2,266,102 | 2,333,853 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,438 | 69,517 | 2,355,220 | 2,424,737 | 0 / 65,536 |
| m4 accumulator | 356,370 | **67,547** | **2,083,860** | **2,151,407** | 0 / 65,536 |
| m4 accumulator stage-fused | 356,392 | 69,263 | 2,203,580 | 2,272,843 | 0 / 65,536 |
| m8 accumulator | **356,226** | 70,397 | 2,191,758 | 2,262,155 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,865 | 68,927 | 2,379,189 | 2,448,116 | 0 / 65,536 |

Fusion makes m2 `1.04x`, m4 `1.06x`, and m8 `1.09x` slower. Baseline m4 is
the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 544,457 | **1,197,143** | 8,164,142 | 9,361,285 | 0 / 65,536 |
| m4 accumulator | **544,254** | 1,197,602 | **7,914,961** | **9,112,563** | 0 / 65,536 |
| m8 accumulator | 544,676 | 1,197,385 | 12,394,883 | 13,592,268 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `4.24x` slower in complete FFT cycles and `3.80x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`4.01x` slower for m2, `4.24x` slower for m4, and `6.01x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,138,357 | 14,438,567 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `6.71x` slower than the best bin-major radix-2
kernel and `1.58x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

At the same VLEN=256, GENV256D64 requires `1.87x` as many best bin-major
butterfly cycles and `1.83x` as many complete FFT cycles as GENV256D128.

#### Complete GENV256D128 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=256, D=128, cold boot, and 10-run
averages. The combined initialization region took 5,122,377 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,744 | 61,341 | 1,395,736 | 1,457,077 | 0 / 65,536 |
| m2 accumulator stage-fused | 358,801 | 61,031 | 1,365,359 | 1,426,390 | 0 / 65,536 |
| m4 accumulator | 356,420 | 61,314 | 1,162,029 | 1,223,343 | 0 / 65,536 |
| m4 accumulator stage-fused | **356,184** | **61,079** | 1,210,619 | 1,271,698 | 0 / 65,536 |
| m8 accumulator | 356,264 | 63,505 | **1,114,775** | **1,178,280** | 0 / 65,536 |
| m8 accumulator stage-fused | 361,596 | 61,520 | 1,310,252 | 1,371,772 | 0 / 65,536 |

Fusion improves m2 by `1.02x`, but makes m4 `1.04x` and m8 `1.18x` slower.
Baseline m8 is the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | **544,082** | **1,195,160** | 7,288,440 | 8,483,600 | 0 / 65,536 |
| m4 accumulator | 544,144 | 1,195,359 | **7,024,485** | **8,219,844** | 0 / 65,536 |
| m8 accumulator | 544,397 | 1,194,905 | 9,612,267 | 10,807,172 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `6.98x` slower in complete FFT cycles and `6.30x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`5.82x` slower for m2, `6.72x` slower for m4, and `9.17x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,690 | 7,579,810 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `6.43x` slower than the best bin-major radix-2
kernel, but `1.08x` faster than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Complete GENV512D64 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=512, D=64, cold boot, and 10-run
averages. The combined initialization region took 5,122,413 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,595 | 47,524 | 2,083,445 | 2,130,969 | 0 / 65,536 |
| m2 accumulator stage-fused | 358,862 | 47,396 | 2,206,151 | 2,253,547 | 0 / 65,536 |
| m4 accumulator | 356,269 | 47,417 | **2,049,658** | **2,097,075** | 0 / 65,536 |
| m4 accumulator stage-fused | 356,303 | 47,459 | 2,118,149 | 2,165,608 | 0 / 65,536 |
| m8 accumulator | 356,303 | 49,280 | 2,322,690 | 2,371,970 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,909 | **47,395** | 2,623,901 | 2,671,296 | 0 / 65,536 |

Fusion makes m2 `1.06x`, m4 `1.03x`, and m8 `1.13x` slower. Baseline m4 is
the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | **544,307** | 1,187,028 | 7,953,064 | 9,140,092 | 0 / 65,536 |
| m4 accumulator | 544,453 | 1,186,960 | **7,852,210** | **9,039,170** | 0 / 65,536 |
| m8 accumulator | 544,464 | **1,186,143** | 16,457,994 | 17,644,137 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `4.31x` slower in complete FFT cycles and `3.83x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`4.29x` slower for m2, `4.31x` slower for m4, and `7.44x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,138,516 | 16,093,798 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `7.67x` slower than the best bin-major radix-2
kernel and `1.78x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

At the same VLEN=512, GENV512D64 requires `1.97x` as many best bin-major
butterfly cycles and `1.94x` as many complete FFT cycles as GENV512D128,
showing the throughput cost of the narrower datapath.

#### Complete GENV512D128 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=512, D=128, cold boot, and 10-run
averages. The combined initialization region took 5,124,717 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,682 | 37,646 | 1,159,181 | 1,196,827 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,318 | 37,808 | 1,210,407 | 1,248,215 | 0 / 65,536 |
| m4 accumulator | 356,244 | 37,648 | **1,041,871** | **1,079,519** | 0 / 65,536 |
| m4 accumulator stage-fused | 356,212 | 37,625 | 1,124,884 | 1,162,509 | 0 / 65,536 |
| m8 accumulator | **356,178** | 39,652 | 1,225,320 | 1,264,972 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,535 | 37,758 | 1,462,279 | 1,500,037 | 0 / 65,536 |

Fusion makes m2 `1.04x`, m4 `1.08x`, and m8 `1.19x` slower. Baseline m4 is
the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 544,256 | 1,183,943 | 7,035,065 | 8,219,008 | 0 / 65,536 |
| m4 accumulator | **544,241** | 1,183,951 | **6,945,834** | **8,129,785** | 0 / 65,536 |
| m8 accumulator | 544,328 | **1,182,719** | 11,619,471 | 12,802,190 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `7.53x` slower in complete FFT cycles and `6.67x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`6.87x` slower for m2, `7.53x` slower for m4, and `10.12x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,721 | 8,318,110 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `7.71x` slower than the best bin-major radix-2
kernel and `1.02x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Complete LGVV128D128 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=128, D=128, cold boot, and 10-run
averages. The combined initialization region took 5,125,874 cycles. All ten
variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,707 | 81,499 | 1,589,855 | 1,671,354 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,450 | 81,410 | 1,720,779 | 1,802,189 | 0 / 65,536 |
| m4 accumulator | 356,142 | **80,493** | **1,294,087** | **1,374,580** | 0 / 65,536 |
| m4 accumulator stage-fused | **356,121** | 80,846 | 1,411,793 | 1,492,639 | 0 / 65,536 |
| m8 accumulator | 356,157 | 83,642 | 1,318,809 | 1,402,451 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,500 | 80,907 | 1,605,037 | 1,685,944 | 0 / 65,536 |

Fusion makes m2 `1.08x`, m4 `1.09x`, and m8 `1.22x` slower. Baseline m4 is
the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | **544,294** | 1,232,148 | 7,397,931 | 8,630,079 | 0 / 65,536 |
| m4 accumulator | 544,345 | **1,231,547** | **6,983,927** | **8,215,474** | 0 / 65,536 |
| m8 accumulator | 544,415 | 1,232,037 | 8,568,471 | 9,800,508 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `5.98x` slower in complete FFT cycles and `5.40x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`5.16x` slower for m2, `5.98x` slower for m4, and `6.99x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,556 | 8,324,841 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `6.06x` slower than the best bin-major radix-2
kernel and `1.01x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Complete LGVV256D128 layout and radix timing data

This Genesys2 FPGA measurement uses VLEN=256, D=128, cold boot, and 10-run
averages. The current initialization region took 5,122,400 cycles and includes
the radix-2 twiddle plan, mixed-radix plan, and batch-major reverse-index
initialization. All ten variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,771 | 61,288 | 1,229,055 | 1,290,343 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,046 | 61,104 | 1,365,249 | 1,426,353 | 0 / 65,536 |
| m4 accumulator | 356,162 | 61,375 | **1,089,147** | **1,150,522** | 0 / 65,536 |
| m4 accumulator stage-fused | 356,225 | 61,100 | 1,208,274 | 1,269,374 | 0 / 65,536 |
| m8 accumulator | 356,197 | 63,498 | 1,114,749 | 1,178,247 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,657 | 61,510 | 1,310,245 | 1,371,755 | 0 / 65,536 |

Fusion makes m2 `1.11x`, m4 `1.11x`, and m8 `1.18x` slower. Baseline m4 is
the fastest bin-major kernel.

##### Radix-2 batch-major `[batch][bin]`

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | **544,207** | 1,195,708 | 6,952,498 | 8,148,206 | 0 / 65,536 |
| m4 accumulator | 544,520 | **1,195,492** | **6,813,326** | **8,008,818** | 0 / 65,536 |
| m8 accumulator | 544,437 | 1,195,996 | 9,434,403 | 10,630,399 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `6.96x` slower in complete FFT cycles and `6.26x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`6.31x` slower for m2, `6.96x` slower for m4, and `9.02x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,619 | 7,579,625 | 69 | 0 / 65,536 | PASS |

The mixed-radix kernel is `6.59x` slower than the best bin-major radix-2
kernel, but `1.06x` faster than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Complete LGVV512D128 timing data

This Genesys2 FPGA measurement uses VLEN=512, D=128, cold boot, and 10-run
averages. It uses the current layout/mixed-radix comparison binary. The
reported initialization region took 5,121,966 cycles and includes the radix-2
twiddle plan, mixed-radix plan, and batch-major reverse-index initialization.
All ten variants passed their corresponding validation.

##### Radix-2 bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 364,675 | 37,576 | 1,099,285 | 1,136,861 | 0 / 65,536 |
| m2 accumulator stage-fused | 359,041 | 37,661 | 1,208,405 | 1,246,066 | 0 / 65,536 |
| m4 accumulator | 356,213 | 37,620 | **1,036,779** | **1,074,399** | 0 / 65,536 |
| m4 accumulator stage-fused | 356,364 | 37,602 | 1,125,477 | 1,163,079 | 0 / 65,536 |
| m8 accumulator | 356,242 | 39,634 | 1,225,322 | 1,264,956 | 0 / 65,536 |
| m8 accumulator stage-fused | 361,594 | 37,686 | 1,446,335 | 1,484,021 | 0 / 65,536 |

Fusion makes m2 `1.10x`, m4 `1.09x`, and m8 `1.18x` slower. Baseline m4 is
the fastest bin-major radix-2 kernel.

##### Radix-2 batch-major `[batch][bin]`

RVV lanes span bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| m2 accumulator | 544,362 | 1,183,694 | 6,821,298 | 8,004,992 | 0 / 65,536 |
| m4 accumulator | 544,459 | 1,183,763 | **6,734,243** | **7,918,006** | 0 / 65,536 |
| m8 accumulator | **544,167** | **1,182,953** | 11,301,730 | 12,484,683 | 0 / 65,536 |

Batch-major m4 is its fastest kernel. Comparing the best radix-2 kernel from
each layout, batch-major is `7.37x` slower in complete FFT cycles and `6.50x`
slower in the butterfly region. At the same baseline LMUL, batch-major is
`7.04x` slower for m2, `7.37x` slower for m4, and `9.87x` slower for m8.

##### Dense mixed-radix 4/16/16 bin-major

| Input layout | FFT | Dense transforms/FFT | Reference mismatches | Result |
| ---: | ---: | ---: | ---: | --- |
| 1,139,813 | 8,316,414 | 69 | 0 / 65,536 | PASS |

The dense mixed-radix kernel is `7.74x` slower than the best bin-major radix-2
kernel and `1.05x` slower than the best batch-major radix-2 kernel. Relative
to radix-2 ground truth, 60,992 / 65,536 points are within +/-1 per component,
the mean absolute component difference is 0.533, and the maximum component
difference is 3.

#### Nine-configuration observations

All nine configurations and all 54 radix-2 bin-major campaign variants passed
bit-exact validation. All nine current extended binaries additionally passed
three batch-major variants and one mixed-radix variant, for ten variants per
configuration. A fused kernel is never the overall winner for a hardware
configuration, even though fusion improves selected m2 or m4 baselines. The
best choices are:

- baseline m8 for `GENV128D64`, `GENV128D128`, and `GENV256D128`;
- baseline m4 for `GENV256D64`, `GENV512D64`, `GENV512D128`,
  `LGVV128D128`, `LGVV256D128`, and `LGVV512D128`.

The fastest hardware result is LGVV512D128 baseline m4 at 1,036,779
butterfly cycles. GENV512D128 baseline m4 is within 0.5% at 1,041,871 cycles.
At VLEN=256, LGVV baseline m8 is only `1.02x` slower than baseline m4,
showing a throughput plateau rather than proportional scaling with LMUL.
These results support
choosing LMUL from the complete dataflow and register pressure, not simply
selecting the largest accumulator LMUL.

Using the current 50 MHz scalar CPU result, baseline RVV provides an `11.05x`
butterfly speedup (`11,461,004 / 1,036,779`) and a `12.85x` complete-FFT
speedup (`13,801,835 / 1,074,399`). When both sides use stage fusion, the best
RVV fused kernel is GENV512D128 m4 fused: it provides a `10.35x` butterfly
speedup (`11,638,084 / 1,124,884`) and an `11.99x` complete-FFT speedup
(`13,940,562 / 1,162,509`). Baseline-to-baseline is the primary hardware
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
