# Batched RVV FFT testbench

This testbench runs 64 independent 1024-point FP32 radix-2 FFTs. Every batch
uses the same input and NumPy ground truth as `testbench/fft`.

## FFT dataflow

The following 8-point radix-2 DIT example shows the same dataflow used by the
1024-point implementation. Input samples are first placed in bit-reversed
order, then processed by $\log_2(8)=3$ butterfly stages. In the batched RVV
implementation, each horizontal wire represents the same FFT bin across all 64
batches, so every butterfly operates on contiguous batch vectors. Red `x`
nodes multiply the lower butterfly input by the labeled twiddle factor, green
`+` nodes form the upper output, and orange `-` nodes form the lower output.
Each purple `B` at a pair of crossing paths marks one complete radix-2
butterfly. The expanded butterfly inset defines $a$ as its upper complex input,
$b$ as its lower complex input, and $t$ as the twiddle-rotated lower input.

For FFT stage $s$, let the block size be $m=2^{s+1}$, the half-block size be
$h=m/2$, the block start be $q$, and the offset inside the half-block be
$k\in[0,h-1]$. If $v_s[n]$ denotes the complex value entering stage $s$, then

$$
\begin{aligned}
a &= v_s[q+k],
&&\text{the upper input of the butterfly}, \\
b &= v_s[q+k+h],
&&\text{the lower input of the butterfly}.
\end{aligned}
$$

Thus, $a$ and $b$ are the two complex values separated by $h$ positions within
the same FFT block. They are not constants. For the 8-point diagram,
$v_0=[x_0,x_4,x_2,x_6,x_1,x_5,x_3,x_7]$ is the bit-reversed input and $v_3$
contains the natural-order FFT outputs $[X_0,\ldots,X_7]$.

The stage-local twiddle factor is

$$
W_m^k = e^{-j 2\pi k/m}.
$$

The diagram expresses every stage using an equivalent power of $W_8$:
$W_m^k=W_8^{8k/m}$.

One butterfly computes

$$
\begin{aligned}
t &= W_m^k b,
&&\text{the twiddle-rotated lower input}, \\
v_{s+1}[q+k] &= a+t,
&&\text{the upper output}, \\
v_{s+1}[q+k+h] &= a-t,
&&\text{the lower output}.
\end{aligned}
$$

In the batched RVV implementation, each scalar-looking symbol is a vector
across the batch dimension:

$$
\mathbf{a}=[a^{(0)},a^{(1)},\ldots,a^{(63)}],\qquad
\mathbf{b}=[b^{(0)},b^{(1)},\ldots,b^{(63)}],\qquad
\mathbf{t}=W_m^k\mathbf{b}.
$$

The twiddle $W_m^k$ is shared by all 64 batches in that butterfly.

![8-point radix-2 DIT FFT dataflow](fft_8point_dataflow.svg)

## Runtime plan and workspace

The caller supplies the input samples, FFT size $N$, and batch count. At
runtime, `fft_plan_init_f32()` verifies that $N$ is a supported power of two,
computes the stage count, and initializes every stage's twiddle table. For a
stage with half-size $h$, its twiddles start at `h - 1`; consequently all
stage-local tables occupy exactly

$$
1+2+4+\cdots+\frac{N}{2}=N-1
$$

complex entries. Twiddle initialization is measured separately as
`twiddle plan cycles` and is not part of `FFT cycles`. A plan may be reused for
multiple inputs of the same shape without recomputing the table.

The implementation allocates four 64-byte-aligned split-complex workspace
buffers through `nn_runtime.h`:

| Buffer | Runtime elements used | Purpose |
| --- | ---: | --- |
| `fft_data_real` | $N\times B$ | real input and in-place real output |
| `fft_data_imag` | $N\times B$ | imaginary input and in-place imaginary output |
| `fft_twiddle_real` | $N-1$ | real part of every stage's twiddles |
| `fft_twiddle_imag` | $N-1$ | imaginary part of every stage's twiddles |

Every allocation is sized from the runtime $N$ and $B$. Linux/Spike links
`csrc/nn_runtime_linux.c`, while bare-metal links
`baremetal/nn_runtime_baremetal.c`; both runtime allocators return 64-byte-
aligned storage. The plan releases all four buffers through `safe_free()` when
the test completes. Thus allocation occurs once during plan initialization and
never inside the hot loop. Input loading writes every normal-order bin into one
contiguous `[bin][batch]` row. At the start of the FFT region, an explicit
`e32,m1` row swap performs bit reversal. The same testbench then runs six
butterfly variants: baseline and independent two-stage-fused kernels for m2,
m4, and m8.
Vector lanes represent batches, so
every vector load/store is unit-stride. Each variant reloads the original
input and repeats bit reversal before it is measured. Results from one LMUL
therefore cannot become input to the next one. This avoids the high FPGA cost
observed for `vsuxei32` while allowing a direct LMUL comparison in one binary.

Cycle reads also go through `nn_runtime_read_cycles()`. Platform selection is
therefore a link-time runtime choice rather than conditional platform code in
the FFT testbench.

Each radix-2 butterfly loads its precomputed scalar twiddle, multiplies the
lower input vector, and computes the upper/lower add-subtract results. The
`offset == 0` case uses
`butterfly_unity_rvv_f32()`, replacing multiplication by $1+0j$ with vector
add/sub operations. Loads, arithmetic results, and stores use separate named
intrinsics rather than nested operands, and the buffer arguments are
`restrict` qualified. This lets the compiler issue independent odd-real,
odd-imaginary, and even-real loads before the dependent multiply/FMA chain;
there are no compiler fences in the butterfly. The normal twiddle path
software-unrolls two independent batch chunks so the compiler can interleave
their loads and arithmetic across the eight available m4 register groups.
Bit-reversal swaps use the FPGA-compatible `e32,m1` sequence. Separate
butterfly functions generate real `e32,m2`, `e32,m4`, and `e32,m8`
instructions; LMUL selection is performed outside every timed hot loop.

The twiddle plan is constructed exactly once and shared by all six variants.
Each variant performs 10 complete runs. A run reloads the original
input, applies bit reversal, and executes every butterfly stage. Reported
layout, bit-reversal, butterfly, and FFT cycle counts are integer averages of
those 10 runs; validation inspects the final run and is outside the timed
regions.

The tuned LMUL=m4 butterfly batch dimension uses a two-chunk unrolled loop, a
single-chunk loop, and at most one tail. The m2 and m8 reference variants use
the same FFT schedule without the two-chunk unroll. The unity-twiddle decision
is outside the block and batch loops in the non-fused variants.

Each `*-stage-fused` variant leaves its original implementation available as a
direct baseline. It processes two consecutive radix-2 stages as one register-blocked
kernel. Four complex input rows produce four first-stage intermediate rows in
LMUL=m2 registers; those values immediately feed the second stage and are
stored only after it completes. This removes one full intermediate store and
reload boundary while retaining each stage's original twiddle lookup and
radix-2 equations. The implementation requires an even stage count, satisfied by the
1,024-point test (`10` stages).

The reported timing regions are (the per-variant values are 10-run averages):

- `twiddle plan cycles`: one-time plan and twiddle initialization;
- `input layout cycles`: contiguous loading into `[bin][batch]`;
- `bit reversal cycles`: only the `e32,m1` row-swap phase;
- `butterfly cycles`: twiddle lookup plus all butterfly stages;
- `FFT cycles`: bit reversal plus butterfly cycles;
- `reused-plan total cycles`: input layout plus FFT;
- `first-run total cycles`: plan initialization, input layout, and FFT.

Stage-fused and non-fused kernels must not be compared by function-call or
loop-iteration count: a fused iteration covers two radix-2 stages. The
`butterfly cycles` timer surrounds the complete 10-stage butterfly transform
for every variant, so it is directly comparable: all variants compute the same
64 FFTs, 10 stage-equivalents, and 327,680 radix-2 butterflies. The testbench
also reports `average butterfly cycles per FFT`. Per-stage-equivalent values
may be obtained by dividing that number by 10, but no timer is inserted inside
the hot loop because its overhead would bias the shorter kernels.

Before entering the measured regions, the testbench rejects NaN/Inf inputs and
checks the conservative FP32 component bound

$$
N\max_n\left(\lvert x_{\mathrm{real}}[n]\rvert
             +\lvert x_{\mathrm{imag}}[n]\rvert\right)
\leq \mathrm{FLT\_MAX}, \qquad N=\mathrm{FFT\_SIZE}.
$$

Each output-bin strip-mined loop also rejects a zero VL or a VL larger than the
remaining bin count. Validation counts NaN/Inf outputs explicitly, so
non-finite arithmetic cannot silently
bypass the normal tolerance comparisons. These checks distinguish arithmetic
overflow and invalid VL behavior from finite-but-incorrect hardware results.
Every one of the $64\times1024=65{,}536$ complex outputs is checked. For a
point outside the configured absolute/relative tolerance, the diagnostic
includes its batch and bin indices, actual and expected complex values, and
component errors. To
keep a failing bare-metal run usable over UART, at most the first 64 mismatches
are printed, followed by the total mismatch count and a mismatch count for each
failing batch. Successful batch-0 bins are not dumped, keeping the final
diagnostic summary visible in captured UART logs.

## Build and run

```sh
./scripts/configure_build.sh linux-pk vector fft_batched zvl128b

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_batched/zvl128b/vector/static/fft_batched
```

Build the bare-metal RVV image:

```sh
make baremetal \
    TESTBENCH=fft_batched \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

Flash it to an SD card:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft_batched \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. The flash target
rejects partitions and mounted devices, then writes the binary beginning at
512-byte block 34. This operation overwrites data on the selected device.

## Performance results

### Current LMUL and m2 stage-fusion Spike result

The following result uses one dynamically generated twiddle plan shared by all
six variants. Each cycle value is the average of 10 runs. Validation passed
with zero mismatched complex points for every variant.

| Variant | Input layout | Bit reversal | Butterfly | FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 360,760 | 221,945 | 1,192,965 | 1,414,910 | 22,107 | 0 / 65,536 |
| `e32,m2-stage-fused` | 336,913 | 221,945 | 813,633 | 1,035,578 | 16,180 | 0 / 65,536 |
| `e32,m4` | 336,912 | 221,944 | 585,745 | 807,689 | 12,620 | 0 / 65,536 |
| `e32,m4-stage-fused` | 336,913 | 221,945 | 442,397 | 664,342 | 10,380 | 0 / 65,536 |
| `e32,m8` | 336,913 | 221,945 | 357,357 | 579,302 | 9,051 | 0 / 65,536 |
| `e32,m8-stage-fused` | 336,913 | 221,945 | 439,843 | 661,788 | 10,340 | 0 / 65,536 |

For the butterfly-only region, stage fusion makes m2 `1.47x` faster and m4
`1.32x` faster than their unchanged baselines. Fused m8 is `1.23x` slower than
baseline m8. Disassembly explains the difference: m2 has no whole-register
spill in its fused kernel, m4 has one `vs4r.v`/`vl4re32.v` spill pair, while m8
has many `vs8r.v`/`vl8re32.v` spills. For m8, spill traffic costs more than the
eliminated stage-boundary load/store traffic.

The m2 input-layout average contains the first execution after plan creation,
whereas the later variants observe warmer cache state. LMUL conclusions should
therefore use the butterfly column; all variants execute the same separate
`e32,m1` bit-reversal implementation.

Twiddle plan creation took 114,100 cycles and is intentionally excluded from
all per-variant FFT averages.

### Genesys2 nine-configuration campaign

All hardware rows use a 50 MHz CPU clock, a 1,024-point FP32 FFT, 64 batches,
one shared twiddle plan, and 10-run averages. A configuration is complete only
after all six variants report `0 / 65,536` mismatches and PASS.

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

#### Main FFT kernel speed table

This is the primary thesis table. Values are average butterfly cycles for the
complete 10-stage transform over 64 FFTs. It excludes input layout, bit
reversal, validation, and one-time twiddle-plan creation. Lower is better.

| Hardware config | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 2,243,617 | 1,905,897 | 1,752,640 | **1,619,368** | 1,656,448 | 2,464,014 | m4 fused |
| `GENV128D128` | 1,881,264 | 1,358,989 | 1,175,968 | 1,012,973 | **893,472** | 1,573,776 | m8 |
| `GENV256D64` | 1,764,814 | 1,611,174 | **1,538,435** | 1,538,590 | 1,675,279 | 2,243,737 | m4 |
| `GENV256D128` | 1,277,344 | 986,395 | 901,643 | 850,794 | **831,941** | 1,246,067 | m8 |
| `GENV512D64` | 1,669,023 | 1,542,146 | 1,679,955 | **1,479,809** | 1,675,761 | 10,690,936 | m4 fused |
| `GENV512D128` | 908,958 | 846,725 | 827,396 | **773,249** | 831,771 | 9,410,351 | m4 fused |
| `LGVV128D128` | 1,769,793 | 1,357,547 | 1,117,048 | 1,006,731 | **853,167** | 1,543,727 | m8 |
| `LGVV256D128` | 1,156,745 | 984,563 | 850,795 | 848,839 | **847,153** | 1,214,008 | m8 |
| `LGVV512D128` | 862,190 | 844,371 | 847,393 | **772,922** | 847,515 | 9,403,442 | m4 fused |

The scalar CPU remains a separate reference row and is not expanded into
`m2/m4/m8` or RVV stage-fusion variants, because it has no LMUL register-group
trade-off. This does not prove that scalar loop fusion could never help; it
means the thesis LMUL/fusion experiment changes only the RVV kernels. CPU/RVV
speedup must use the current scalar FFT's equivalently scoped butterfly cycles,
which should be rerun with the same 10-run averaging policy before the final
comparison table is published.

#### Complete GENV128D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,496 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 463,060 | 321,708 | 2,243,617 | 2,565,325 | 35,056 | 40,083 | 0 / 65,536 |
| `e32,m2-stage-fused` | 455,082 | 315,384 | 1,905,897 | 2,221,281 | 29,779 | 34,707 | 0 / 65,536 |
| `e32,m4` | 443,749 | 315,493 | 1,752,640 | 2,068,133 | 27,385 | 32,314 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,983 | 315,380 | 1,619,368 | 1,934,748 | 25,302 | 30,230 | 0 / 65,536 |
| `e32,m8` | 454,871 | 314,210 | 1,656,448 | 1,970,658 | 25,882 | 30,791 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,141 | 315,896 | 2,464,014 | 2,779,910 | 38,500 | 43,436 | 0 / 65,536 |

On GENV128D64, stage fusion improves m2 by `1.18x` and m4 by `1.08x`, but
makes m8 `1.49x` slower. Fused m4 is the best kernel and is `1.02x` faster
than baseline m8. Comparing the best kernels at the same VLEN, GENV128D64
requires `1.81x` as many butterfly cycles as GENV128D128.

#### Complete GENV128D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,634 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,341 | 280,467 | 1,881,264 | 2,161,731 | 29,394 | 33,777 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,672 | 273,669 | 1,358,989 | 1,632,658 | 21,234 | 25,510 | 0 / 65,536 |
| `e32,m4` | 444,492 | 274,132 | 1,175,968 | 1,450,100 | 18,374 | 22,657 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,775 | 273,529 | 1,012,973 | 1,286,502 | 15,827 | 20,101 | 0 / 65,536 |
| `e32,m8` | 454,785 | 272,334 | 893,472 | 1,165,806 | 13,960 | 18,215 | 0 / 65,536 |
| `e32,m8-stage-fused` | 454,850 | 274,026 | 1,573,776 | 1,847,802 | 24,590 | 28,871 | 0 / 65,536 |

On GENV128D128, stage fusion improves m2 by `1.38x` and m4 by `1.16x`, but
makes m8 `1.76x` slower. Baseline m8 is the best kernel and is `1.13x` faster
than fused m4. Its best-kernel butterfly count is about 5% higher than
LGVV128D128's baseline m8 result.

#### Complete GENV256D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,217 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 463,034 | 243,167 | 1,764,814 | 2,007,981 | 27,575 | 31,374 | 0 / 65,536 |
| `e32,m2-stage-fused` | 455,122 | 236,998 | 1,611,174 | 1,848,172 | 25,174 | 28,877 | 0 / 65,536 |
| `e32,m4` | 443,770 | 236,879 | 1,538,435 | 1,775,314 | 24,038 | 27,739 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,880 | 237,044 | 1,538,590 | 1,775,634 | 24,040 | 27,744 | 0 / 65,536 |
| `e32,m8` | 454,828 | 235,407 | 1,675,279 | 1,910,686 | 26,176 | 29,854 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,383 | 238,278 | 2,243,737 | 2,482,015 | 35,058 | 38,781 | 0 / 65,536 |

On GENV256D64, stage fusion improves m2 by `1.10x`, while m4 fusion is
effectively neutral and is 155 cycles slower than its baseline. Baseline m4 is
the best kernel. Fused m8 is `1.34x` slower than baseline m8. Comparing the
best kernels at the same VLEN, GENV256D64 requires `1.85x` as many butterfly
cycles as GENV256D128, showing the throughput cost of the narrower datapath.

#### Complete GENV256D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,394 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,104 | 205,477 | 1,277,344 | 1,482,821 | 19,958 | 23,169 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,624 | 199,354 | 986,395 | 1,185,749 | 15,412 | 18,527 | 0 / 65,536 |
| `e32,m4` | 444,347 | 199,545 | 901,643 | 1,101,188 | 14,088 | 17,206 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,913 | 199,069 | 850,794 | 1,049,863 | 13,293 | 16,404 | 0 / 65,536 |
| `e32,m8` | 454,801 | 198,292 | 831,941 | 1,030,233 | 12,999 | 16,097 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,130 | 200,371 | 1,246,067 | 1,446,438 | 19,469 | 22,600 | 0 / 65,536 |

On GENV256D128, stage fusion improves m2 by `1.29x` and m4 by `1.06x`, but
makes m8 `1.50x` slower. Baseline m8 is the best kernel and is `1.02x` faster
than fused m4. As on LGVV256D128, the VLEN=256/D=128 configuration places
baseline m8 and fused m4 very close together, while the fused m8 register
pressure outweighs the eliminated stage-boundary traffic.

#### Complete GENV512D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,327 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 462,990 | 212,642 | 1,669,023 | 1,881,665 | 26,078 | 29,401 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,910 | 207,791 | 1,542,146 | 1,749,937 | 24,096 | 27,342 | 0 / 65,536 |
| `e32,m4` | 443,783 | 207,792 | 1,679,955 | 1,887,747 | 26,249 | 29,496 | 0 / 65,536 |
| `e32,m4-stage-fused` | 455,191 | 207,812 | 1,479,809 | 1,687,621 | 23,122 | 26,369 | 0 / 65,536 |
| `e32,m8` | 454,915 | 206,941 | 1,675,761 | 1,882,702 | 26,183 | 29,417 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,774 | 208,969 | 10,690,936 | 10,899,905 | 167,045 | 170,311 | 0 / 65,536 |

On GENV512D64, stage fusion improves m2 by `1.08x` and m4 by `1.14x`, while
fused m8 is `6.38x` slower than baseline m8. Fused m4 is the best kernel and
is `1.13x` faster than baseline m8. The three non-fused LMUL baselines differ
by less than 1%, showing that the D64 datapath dominates their throughput.
Compared with GENV512D128, D64 roughly doubles the useful baseline/fused-m4
cycles, while the m8 spill penalty remains severe.

#### Complete GENV512D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,427 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,239 | 158,133 | 908,958 | 1,067,091 | 14,202 | 16,673 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,698 | 153,745 | 846,725 | 1,000,470 | 13,230 | 15,632 | 0 / 65,536 |
| `e32,m4` | 444,147 | 153,836 | 827,396 | 981,232 | 12,928 | 15,331 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,825 | 153,773 | 773,249 | 927,022 | 12,082 | 14,484 | 0 / 65,536 |
| `e32,m8` | 454,572 | 152,458 | 831,771 | 984,229 | 12,996 | 15,378 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,544 | 155,023 | 9,410,351 | 9,565,374 | 147,036 | 149,458 | 0 / 65,536 |

On GENV512D128, stage fusion improves m2 by `1.07x` and m4 by `1.07x`, while
fused m8 is `11.31x` slower than baseline m8. Fused m4 is the best kernel and
is `1.08x` faster than baseline m8. The result closely matches LGVV512D128:
both D128/VLEN=512 configurations favor m4 fusion and expose a roughly 11x m8
spill penalty.

#### Complete LGVV128D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,398 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,597 | 280,754 | 1,769,793 | 2,050,547 | 27,653 | 32,039 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,706 | 273,557 | 1,357,547 | 1,631,104 | 21,211 | 25,486 | 0 / 65,536 |
| `e32,m4` | 444,323 | 274,158 | 1,117,048 | 1,391,206 | 17,453 | 21,737 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,784 | 273,519 | 1,006,731 | 1,280,250 | 15,730 | 20,003 | 0 / 65,536 |
| `e32,m8` | 454,683 | 272,587 | 853,167 | 1,125,754 | 13,330 | 17,589 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,207 | 274,010 | 1,543,727 | 1,817,737 | 24,120 | 28,402 | 0 / 65,536 |

On LGVV128D128, stage fusion improves m2 by `1.30x` and m4 by `1.11x`, but
makes m8 `1.81x` slower. Baseline m8 is the best kernel and is `1.18x` faster
than fused m4. With VLEN=128, the 64-batch row requires eight m2 chunks, four
m4 chunks, or two m8 chunks; the reduced instruction/chunk count makes
baseline m8 beneficial despite the fixed D128 datapath, while fused m8 again
suffers from whole-register spills.

#### Complete LGVV256D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. Twiddle plan creation took
173,353 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,339 | 205,729 | 1,156,745 | 1,362,474 | 18,074 | 21,288 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,808 | 199,062 | 984,563 | 1,183,625 | 15,383 | 18,494 | 0 / 65,536 |
| `e32,m4` | 444,561 | 199,247 | 850,795 | 1,050,042 | 13,293 | 16,406 | 0 / 65,536 |
| `e32,m4-stage-fused` | 455,030 | 199,703 | 848,839 | 1,048,542 | 13,263 | 16,383 | 0 / 65,536 |
| `e32,m8` | 454,771 | 197,905 | 847,153 | 1,045,058 | 13,236 | 16,329 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,030 | 200,121 | 1,214,008 | 1,414,129 | 18,968 | 22,095 | 0 / 65,536 |

On LGVV256D128, stage fusion improves m2 by `1.17x`, changes m4 by only
`1.002x`, and makes m8 `1.43x` slower. Baseline m8 is the best kernel, although
it is only `1.002x` faster than fused m4. At VLEN=256, m4 uses VL=32 and needs
two batch chunks, whereas m8 uses VL=64 and covers the batch row once; the
near-equal cycles show that larger instruction coverage does not translate
into proportional throughput on this D128 hardware. Fused m8 still suffers
heavy register spilling.

#### Complete LGVV512D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=128, cold boot,
one shared twiddle plan, and 10 runs per LMUL variant. Twiddle plan creation
took 173,443 cycles. All six variants passed validation.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,102 | 158,021 | 862,190 | 1,020,211 | 13,471 | 15,940 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,545 | 153,760 | 844,371 | 998,131 | 13,193 | 15,595 | 0 / 65,536 |
| `e32,m4` | 444,314 | 154,118 | 847,393 | 1,001,511 | 13,240 | 15,648 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,927 | 153,485 | 772,922 | 926,407 | 12,076 | 14,475 | 0 / 65,536 |
| `e32,m8` | 454,491 | 152,842 | 847,515 | 1,000,357 | 13,242 | 15,630 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,401 | 155,144 | 9,403,442 | 9,558,586 | 146,928 | 149,352 | 0 / 65,536 |

The three non-fused baselines remain nearly identical on LGVV512D128. Stage
fusion improves m2 by only `1.02x`, but improves m4 by `1.10x`; fused m4 is the
fastest measured hardware kernel and is `1.10x` faster than baseline m8. In
contrast, fused m8 is `11.10x` slower than baseline m8. The hardware penalty is
far larger than Spike's `1.23x` slowdown and is consistent with the many m8
whole-register spills seen in disassembly. This result demonstrates that the
cost of spill/reload traffic is strongly microarchitecture-dependent and that
the largest LMUL is not optimal once multiple FFT stages are kept live.

### Archived earlier results

> The recorded rows below predate the hot-loop branch hoist and separate
> bit-reversal / butterfly counters. Re-run the current binaries before using
> these values for the final CPU/RVV comparison.

The rows below describe the immediately preceding LMUL=m8-only binary. The
current four-buffer implementation prints separate `RVV variant: m2`, `m4`,
and `m8` sections from one execution. It uses contiguous input layout,
`e32,m1` bit-reversal row swaps, and no unity decision in the block/batch hot
loops. Re-run it to populate comparable three-variant measurements.

The Spike command for the current implementation is:

```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_batched/zvl128b/vector/static/fft_batched
```

| Environment | Configuration | FFT size | Batches | Twiddle plan cycles | Input layout cycles | FFT cycles | Reused-plan total cycles | First-run total cycles | FFT cycles/FFT | Reused-plan cycles/FFT | First-run cycles/FFT | Mismatches | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | Scalar CPU | 1,024 | 64 | 114,235 | 839,015 | 15,432,206 | 16,271,221 | 16,385,456 | 241,128 | 254,237 | 256,022 | 0 / 65,536 | PASS |
| Spike simulation | `zvl128b`, LMUL=m8, contiguous layout + row swap | 1,024 | 64 | 105,900 | 576,403 | 517,852 | 1,094,255 | 1,200,155 | 8,091 | 17,097 | 18,752 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | Scalar CPU, 50 MHz | 1,024 | 64 | 171,198 | 759,907 | 12,760,529 | 13,520,436 | 13,691,634 | 199,383 | 211,256 | 213,931 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV128D64`, VLEN=128, LMUL=m8 | 1,024 | 64 | 171,159 | 544,211 | 2,129,139 | 2,673,350 | 2,844,509 | 33,267 | 41,771 | 44,445 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV128D128`, VLEN=128, LMUL=m8 | 1,024 | 64 | 171,095 | 545,892 | 1,309,551 | 1,855,443 | 2,026,538 | 20,461 | 28,991 | 31,664 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV256D64`, VLEN=256, LMUL=m8 | 1,024 | 64 | 171,414 | 543,767 | 2,033,171 | 2,576,938 | 2,748,352 | 31,768 | 40,264 | 42,943 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV256D128`, VLEN=256, LMUL=m8 | 1,024 | 64 | 171,275 | 545,437 | 1,175,785 | 1,721,222 | 1,892,497 | 18,371 | 26,894 | 29,570 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `GENV512D64`, VLEN=512, LMUL=m8 | 1,024 | 64 | 171,371 | 545,571 | 1,110,281 | 1,655,852 | 1,827,223 | 17,348 | 25,872 | 28,550 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV128D128`, VLEN=128, LMUL=m8 | 1,024 | 64 | 171,119 | 546,616 | 1,276,698 | 1,823,314 | 1,994,433 | 19,948 | 28,489 | 31,163 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV256D128`, VLEN=256, LMUL=m8 | 1,024 | 64 | 171,217 | 545,702 | 1,136,012 | 1,681,714 | 1,852,931 | 17,750 | 26,276 | 28,952 | 0 / 65,536 | PASS |
| Genesys2 FPGA, cold cache | `LGVV512D128`, VLEN=512, LMUL=m8 | 1,024 | 64 | 171,119 | 546,616 | 1,064,237 | 1,610,853 | 1,781,972 | 16,628 | 25,169 | 27,843 | 0 / 65,536 | PASS |

For the RVV rows, the maximum component error was `0.000026`, and the maximum
tolerance ratio was `0.461186`; both occurred at batch 0, bin 1015. The scalar
CPU rows reported maximum component error `0.000024` at batch 0, bin 757 and
maximum tolerance ratio `0.385719` at batch 0, bin 1015. No non-finite outputs
were observed. `First-run total` includes allocation and twiddle initialization,
whereas `reused-plan total` represents another input using an already-created
plan and excludes that row's one-time plan cost.

### Spike speedup over scalar CPU

The scalar and RVV implementations both use normal-order contiguous input
layout and perform bit reversal inside the measured FFT region. Each value is
from one Spike run rather than a multi-run average.

| Compared region | Scalar CPU cycles | RVV cycles | RVV speedup |
| --- | ---: | ---: | ---: |
| FFT kernel | 15,432,206 | 517,852 | 29.80x |
| Reused-plan total | 16,271,221 | 1,094,255 | 14.87x |
| First-run total | 16,385,456 | 1,200,155 | 13.65x |

Both implementations process 64 independent 1024-point FFTs and pass all
65,536 complex-point checks. The kernel comparison excludes plan and input
layout costs; the reused-plan comparison includes input layout; the first-run
comparison additionally includes allocation and twiddle initialization.

### Genesys2 speedup over scalar CPU

The scalar CPU baseline is the matching 50 MHz Genesys2 cold-cache row in the
main table above.

`Max component error` and `Max tolerance ratio` depend on the operator sequence
used by each implementation. Separate multiply/add operations, fused
multiply-add/subtract instructions, unity-twiddle specialization, and a changed
evaluation order can produce different valid FP32 rounding results. These
values therefore describe numerical margin rather than rank implementations by
correctness. The validation result is determined by the number of points whose
tolerance ratio exceeds 1.0, together with the non-finite output count; all
results compared here have zero mismatches and zero non-finite outputs.

For component $c\in\{\mathrm{real},\mathrm{imag}\}$, the reported tolerance
ratio is

$$
r_c = \frac{\lvert y_c-\hat{y}_c\rvert}
           {\mathrm{atol}+\mathrm{rtol}\lvert\hat{y}_c\rvert},
\qquad
r=\max(r_{\mathrm{real}},r_{\mathrm{imag}}).
$$

A complex point is a mismatch when $r>1$.

The speedups in the table are calculated as

$$
S_{\mathrm{kernel}}
=\frac{C_{\mathrm{CPU/FFT}}}{C_{\mathrm{RVV\ kernel/FFT}}},
\qquad
S_{\mathrm{reused}}
=\frac{C_{\mathrm{CPU/FFT}}}{C_{\mathrm{RVV\ reused/FFT}}},
\qquad
S_{\mathrm{first}}
=\frac{C_{\mathrm{CPU/FFT}}}{C_{\mathrm{RVV\ first/FFT}}}.
$$

| RVV configuration | FFT cycles/FFT | Reused cycles/FFT | First-run cycles/FFT | Kernel speedup | Reused-plan speedup | First-run speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GENV128D64` | 33,267 | 41,771 | 44,445 | 5.99x | 5.06x | 4.81x |
| `GENV128D128` | 20,461 | 28,991 | 31,664 | 9.74x | 7.29x | 6.76x |
| `GENV256D64` | 31,768 | 40,264 | 42,943 | 6.28x | 5.25x | 4.98x |
| `GENV256D128` | 18,371 | 26,894 | 29,570 | 10.85x | 7.86x | 7.23x |
| `GENV512D64` | 17,348 | 25,872 | 28,550 | 11.49x | 8.17x | 7.49x |
| `LGVV128D128` | 19,948 | 28,489 | 31,163 | 9.99x | 7.42x | 6.86x |
| `LGVV256D128` | 17,750 | 26,276 | 28,952 | 11.23x | 8.04x | 7.39x |
| `LGVV512D128` | 16,628 | 25,169 | 27,843 | 11.99x | 8.39x | 7.68x |
