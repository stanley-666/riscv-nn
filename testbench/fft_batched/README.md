<!-- SPDX-FileContributor: Person: Stanley Lee -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Batched RVV FFT testbench

Linux/Spike and bare-metal compile the same
`testbench/fft_batched/fft_batched.c` source and execute the same `main()`.
Bare-metal replaces only startup, linking, minilib, and the `nn_runtime`
implementation; there is no separate application adapter.
The bare-metal startup prints hardware and RVV/VLEN information before entering
this shared `main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

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

The twiddle plan is constructed exactly once and shared by all six bin-major
variants and all three batch-major variants. Each variant performs 10 complete
runs. A run reloads the original
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

### FP32 bin-major versus batch-major Spike result

The same binary also measures an explicit `[batch][bin]` layout. In the
original `[bin][batch]` kernels, vector lanes span independent batches and
each butterfly uses a shared scalar twiddle. In the batch-major kernels,
vector lanes span consecutive FFT bins and the corresponding twiddles are
loaded as vectors. All rows below process 64 independent 1,024-point FFTs,
average 10 runs, and pass all 65,536 complex-point comparisons.

| Layout | Variant | Input layout | Bit reversal | Butterfly | FFT | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bin-major | `m2` | 360,760 | 221,945 | 1,192,965 | 1,414,910 | 0 / 65,536 |
| Bin-major | `m4` | 336,928 | 230,881 | 662,271 | 893,152 | 0 / 65,536 |
| Bin-major | `m8` | 336,913 | 221,945 | 357,357 | 579,302 | 0 / 65,536 |
| Batch-major | `m2` | 590,235 | 4,968,994 | 4,216,406 | 9,185,400 | 0 / 65,536 |
| Batch-major | `m4` | 590,235 | 4,969,061 | 2,940,206 | 7,909,267 | 0 / 65,536 |
| Batch-major | `m8` | 590,235 | 4,968,996 | 4,504,727 | 9,473,723 | 0 / 65,536 |

Batch-major `m4` is its fastest LMUL, but its butterfly region is `8.23x`
slower than bin-major `m8`, and its complete FFT region is `13.65x` slower.
The batch-major layout must perform a separate bit-reversal permutation for
each FFT, while bin-major swaps two contiguous 64-element batch rows. It also
loads a vector of twiddles instead of broadcasting one twiddle across all
batches. These results therefore favor bin-major for this 64-batch workload.

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
one shared twiddle plan, and 10-run averages. A current layout-comparison
configuration is complete only after all nine variants report
`0 / 65,536` mismatches and PASS.

| Hardware config | VLEN | Datapath | Complete measured result |
| --- | ---: | ---: | --- |
| `GENV128D64` | 128 | 64 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `GENV128D128` | 128 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `GENV256D64` | 256 | 64 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `GENV256D128` | 256 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `GENV512D64` | 512 | 64 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `GENV512D128` | 512 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `LGVV128D128` | 128 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `LGVV256D128` | 256 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |
| `LGVV512D128` | 512 | 128 | PASS, 0 / 65,536 for all 9 layout/LMUL variants |

#### Main FFT kernel speed table

This is the primary thesis table. Values are average butterfly cycles for the
complete 10-stage transform over 64 FFTs. It excludes input layout, bit
reversal, validation, and one-time twiddle-plan creation. Lower is better.

| Hardware config | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 2,241,998 | 1,904,823 | 1,856,716 | **1,618,475** | 1,660,562 | 2,466,670 | m4 fused |
| `GENV128D128` | 1,880,734 | 1,358,932 | 1,337,759 | 1,013,401 | **896,622** | 1,574,612 | m8 |
| `GENV256D64` | 1,765,029 | 1,610,780 | 1,585,568 | **1,537,331** | 1,679,471 | 2,245,775 | m4 fused |
| `GENV256D128` | 1,276,149 | 987,527 | 1,018,141 | 850,743 | **835,454** | 1,247,631 | m8 |
| `GENV512D64` | 1,673,672 | 1,540,598 | 1,681,172 | **1,479,617** | 1,678,834 | 6,719,823 | m4 fused |
| `GENV512D128` | 911,399 | 845,312 | 829,309 | **773,508** | 835,376 | 5,265,212 | m4 fused |
| `LGVV128D128` | 1,767,700 | 1,358,129 | 1,279,325 | 1,007,728 | **852,932** | 1,545,075 | m8 |
| `LGVV256D128` | 1,157,176 | 985,849 | 970,153 | **847,929** | 853,838 | 1,214,280 | m4 fused |
| `LGVV512D128` | 861,070 | 842,949 | 828,128 | **773,128** | 854,032 | 5,316,059 | m4 fused |

The scalar CPU remains a separate reference row and is not expanded into
`m2/m4/m8` or RVV stage-fusion variants, because it has no LMUL register-group
trade-off. This does not prove that scalar loop fusion could never help; it
means the thesis LMUL/fusion experiment changes only the RVV kernels. CPU/RVV
speedup must use the current scalar FFT's equivalently scoped butterfly cycles,
which should be rerun with the same 10-run averaging policy before the final
comparison table is published.

#### Complete GENV128D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. The supplied UART excerpt starts at the bin-major
heading, so it does not contain the twiddle-plan cycle line. All nine variants
passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 462,937 | 321,659 | 2,241,998 | 2,563,657 | 35,031 | 40,057 | 0 / 65,536 |
| `e32,m2-stage-fused` | 455,037 | 315,007 | 1,904,823 | 2,219,830 | 29,762 | 34,684 | 0 / 65,536 |
| `e32,m4` | 453,574 | 334,994 | 1,856,716 | 2,191,710 | 29,011 | 34,245 | 0 / 65,536 |
| `e32,m4-stage-fused` | 455,064 | 315,324 | **1,618,475** | **1,933,799** | **25,288** | **30,215** | 0 / 65,536 |
| `e32,m8` | 454,969 | **314,187** | 1,660,562 | 1,974,749 | 25,946 | 30,855 | 0 / 65,536 |
| `e32,m8-stage-fused` | 454,940 | 315,604 | 2,466,670 | 2,782,274 | 38,541 | 43,473 | 0 / 65,536 |

Stage fusion improves m2 by `1.18x` and m4 by `1.15x`, while fused m8 is
`1.49x` slower than baseline m8. Fused m4 is the fastest bin-major kernel.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 758,174 | **3,979,053** | 8,858,914 | **12,837,967** | 138,420 | **200,593** | 0 / 65,536 |
| `e32,m4` | 761,704 | 4,506,700 | **8,488,551** | 12,995,251 | **132,633** | 203,050 | 0 / 65,536 |
| `e32,m8` | **757,542** | 3,997,235 | 13,728,894 | 17,726,129 | 214,513 | 276,970 | 0 / 65,536 |

Batch-major m2 is its best complete FFT, while m4 has its best butterfly
region. Comparing the best complete kernels, batch-major is `6.64x` slower
than bin-major. Its best butterfly region is `5.24x` slower. At the same
non-fused LMUL, batch-major is `5.01x` slower for m2, `5.93x` slower for m4,
and `8.98x` slower for m8 in complete FFT cycles.

At the same VLEN=128, GENV128D64 requires `1.81x` as many best-kernel
butterfly cycles and `1.65x` as many complete FFT cycles as GENV128D128.

#### Complete GENV128D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. The supplied UART excerpt starts at the bin-major
heading, so it does not contain the twiddle-plan cycle line. All nine variants
passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,294 | 281,001 | 1,880,734 | 2,161,735 | 29,386 | 33,777 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,841 | 273,428 | 1,358,932 | 1,632,360 | 21,233 | 25,505 | 0 / 65,536 |
| `e32,m4` | 453,908 | 294,142 | 1,337,759 | 1,631,901 | 20,902 | 25,498 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,690 | 273,324 | 1,013,401 | 1,286,725 | 15,834 | 20,105 | 0 / 65,536 |
| `e32,m8` | 454,924 | **272,596** | **896,622** | **1,169,218** | **14,009** | **18,269** | 0 / 65,536 |
| `e32,m8-stage-fused` | 454,695 | 273,975 | 1,574,612 | 1,848,587 | 24,603 | 28,884 | 0 / 65,536 |

Stage fusion improves m2 by `1.38x` and m4 by `1.32x`, but makes m8 `1.76x`
slower. Baseline m8 remains the fastest bin-major kernel. Its complete FFT is
approximately `1.04x` slower than LGVV128D128's baseline m8 result.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 761,981 | **3,995,877** | 8,423,113 | 12,418,990 | 131,611 | 194,046 | 0 / 65,536 |
| `e32,m4` | 765,565 | 4,523,745 | **7,844,723** | **12,368,468** | **122,573** | **193,257** | 0 / 65,536 |
| `e32,m8` | **759,799** | 4,014,883 | 10,997,182 | 15,012,065 | 171,830 | 234,563 | 0 / 65,536 |

Batch-major m4 is its fastest butterfly and complete FFT kernel. Comparing the
best complete kernels, batch-major is `10.58x` slower than bin-major. Its best
butterfly region is `8.75x` slower. At the same non-fused LMUL, batch-major is
`5.74x` slower for m2, `7.58x` slower for m4, and `12.84x` slower for m8 in
complete FFT cycles.

#### Complete GENV256D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 191,888 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 463,119 | 243,168 | 1,765,029 | 2,008,197 | 27,578 | 31,378 | 0 / 65,536 |
| `e32,m2-stage-fused` | 455,083 | 237,093 | 1,610,780 | 1,847,873 | 25,168 | 28,873 | 0 / 65,536 |
| `e32,m4` | 453,859 | 256,462 | 1,585,568 | 1,842,030 | 24,774 | 28,781 | 0 / 65,536 |
| `e32,m4-stage-fused` | 455,299 | 236,899 | **1,537,331** | **1,774,230** | **24,020** | **27,722** | 0 / 65,536 |
| `e32,m8` | 454,913 | **235,397** | 1,679,471 | 1,914,868 | 26,241 | 29,919 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,315 | 237,736 | 2,245,775 | 2,483,511 | 35,090 | 38,804 | 0 / 65,536 |

Stage fusion improves m2 by `1.10x` and m4 by `1.03x`, while fused m8 is
`1.34x` slower than baseline m8. Fused m4 is the fastest bin-major kernel.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | **757,574** | **3,979,025** | 8,259,873 | **12,238,898** | 129,060 | **191,232** | 0 / 65,536 |
| `e32,m4` | 762,288 | 4,506,614 | **8,240,695** | 12,747,309 | **128,760** | 199,176 | 0 / 65,536 |
| `e32,m8` | 757,844 | 3,998,686 | 17,937,654 | 21,936,340 | 280,275 | 342,755 | 0 / 65,536 |

Batch-major m2 is its best complete FFT, while m4 has its best butterfly
region. Comparing the best complete kernels, batch-major is `6.90x` slower
than bin-major. Its best butterfly region is `5.36x` slower. At the same
non-fused LMUL, batch-major is `6.09x` slower for m2, `6.92x` slower for m4,
and `11.46x` slower for m8 in complete FFT cycles.

At the same VLEN=256, GENV256D64 requires `1.84x` as many best-kernel
butterfly cycles and `1.72x` as many complete FFT cycles as GENV256D128,
showing the throughput cost of the narrower datapath.

#### Complete GENV256D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 191,961 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,389 | 205,694 | 1,276,149 | 1,481,843 | 19,939 | 23,153 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,669 | 199,234 | 987,527 | 1,186,761 | 15,430 | 18,543 | 0 / 65,536 |
| `e32,m4` | 453,861 | 219,170 | 1,018,141 | 1,237,311 | 15,908 | 19,332 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,651 | 199,609 | 850,743 | 1,050,352 | 13,292 | 16,411 | 0 / 65,536 |
| `e32,m8` | 454,662 | **198,117** | **835,454** | **1,033,571** | **13,053** | **16,149** | 0 / 65,536 |
| `e32,m8-stage-fused` | 454,985 | 200,483 | 1,247,631 | 1,448,114 | 19,494 | 22,626 | 0 / 65,536 |

Stage fusion improves m2 by `1.29x` and m4 by `1.20x`, but makes m8 `1.49x`
slower. Baseline m8 is the fastest bin-major kernel and is `1.02x` faster than
fused m4. This remains consistent with LGVV256D128, whose fused-m4 result is
within approximately 1.3% of this best complete FFT.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 761,198 | **3,995,629** | 7,690,714 | **11,686,343** | 120,167 | **182,599** | 0 / 65,536 |
| `e32,m4` | 764,948 | 4,521,549 | **7,594,516** | 12,116,065 | **118,664** | 189,313 | 0 / 65,536 |
| `e32,m8` | **759,705** | 4,014,570 | 12,661,042 | 16,675,612 | 197,828 | 260,556 | 0 / 65,536 |

Batch-major m2 is its best complete FFT, while m4 has its best butterfly
region. Comparing the best complete kernels, batch-major is `11.31x` slower
than bin-major. Its best butterfly region is `9.09x` slower. At the same
non-fused LMUL, batch-major is `7.89x` slower for m2, `9.79x` slower for m4,
and `16.13x` slower for m8 in complete FFT cycles.

#### Complete GENV512D64 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=64, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 191,947 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 463,015 | 212,756 | 1,673,672 | 1,886,428 | 26,151 | 29,475 | 0 / 65,536 |
| `e32,m2-stage-fused` | 455,031 | 207,680 | 1,540,598 | 1,748,278 | 24,071 | 27,316 | 0 / 65,536 |
| `e32,m4` | 453,916 | 227,765 | 1,681,172 | 1,908,937 | 26,268 | 29,827 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,912 | 207,776 | **1,479,617** | **1,687,393** | **23,119** | **26,365** | 0 / 65,536 |
| `e32,m8` | 454,995 | **206,888** | 1,678,834 | 1,885,722 | 26,231 | 29,464 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,851 | 208,806 | 6,719,823 | 6,928,629 | 104,997 | 108,259 | 0 / 65,536 |

Stage fusion improves m2 by `1.09x` and m4 by `1.14x`, while fused m8 is
`4.00x` slower than baseline m8. Fused m4 is the fastest bin-major kernel.
The three non-fused LMUL baselines remain within approximately 1%, showing
that the 64-bit datapath dominates their throughput.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 758,150 | **3,980,238** | **8,127,535** | **12,107,773** | **126,992** | **189,183** | 0 / 65,536 |
| `e32,m4` | 763,322 | 4,506,248 | 8,246,211 | 12,752,459 | 128,847 | 199,257 | 0 / 65,536 |
| `e32,m8` | **757,747** | 3,997,883 | 27,295,431 | 31,293,314 | 426,491 | 488,958 | 0 / 65,536 |

Batch-major m2 is its best kernel at 12,107,773 complete FFT cycles. Comparing
the best complete kernels, batch-major is `7.18x` slower than bin-major. Its
best butterfly region is `5.49x` slower. At the same non-fused LMUL,
batch-major is `6.42x` slower for m2, `6.68x` slower for m4, and `16.59x`
slower for m8 in complete FFT cycles.

Compared with GENV512D128, D64 roughly doubles the useful bin-major
baseline/fused-m4 cycles. Its batch-major m8 is also especially expensive,
consistent with spill traffic traversing the narrower datapath.

#### Complete GENV512D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 191,955 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,384 | 158,005 | 911,399 | 1,069,404 | 14,240 | 16,709 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,505 | 153,583 | 845,312 | 998,895 | 13,208 | 15,607 | 0 / 65,536 |
| `e32,m4` | 454,102 | 173,327 | 829,309 | 1,002,636 | 12,957 | 15,666 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,984 | 153,628 | **773,508** | **927,136** | **12,086** | **14,486** | 0 / 65,536 |
| `e32,m8` | 454,734 | **152,576** | 835,376 | 987,952 | 13,052 | 15,436 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,515 | 154,989 | 5,265,212 | 5,420,201 | 82,268 | 84,690 | 0 / 65,536 |

Stage fusion improves m2 by `1.08x` and m4 by `1.07x`, while fused m8 is
`6.30x` slower than baseline m8. Fused m4 is the fastest bin-major kernel.
Both GENV512D128 and LGVV512D128 therefore favor m4 fusion and expose a severe
m8 spill penalty (`6.30x` and `6.22x`, respectively).

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 760,903 | **3,995,457** | **7,439,795** | **11,435,252** | **116,246** | **178,675** | 0 / 65,536 |
| `e32,m4` | 766,423 | 4,523,880 | 7,607,431 | 12,131,311 | 118,866 | 189,551 | 0 / 65,536 |
| `e32,m8` | **759,753** | 4,015,217 | 16,919,743 | 20,934,960 | 264,370 | 327,108 | 0 / 65,536 |

Batch-major m2 is its best kernel at 11,435,252 complete FFT cycles. Comparing
the best complete kernels, batch-major is `12.33x` slower than bin-major. Its
best butterfly region is `9.62x` slower. At the same non-fused LMUL,
batch-major is `10.69x` slower for m2, `12.10x` slower for m4, and `21.19x`
slower for m8 in complete FFT cycles.

#### Complete LGVV128D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=128, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 192,070 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,259 | 280,649 | 1,767,700 | 2,048,349 | 27,620 | 32,005 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,721 | 273,342 | 1,358,129 | 1,631,471 | 21,220 | 25,491 | 0 / 65,536 |
| `e32,m4` | 454,099 | 293,989 | 1,279,325 | 1,573,314 | 19,989 | 24,583 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,482 | 273,641 | 1,007,728 | 1,281,369 | 15,745 | 20,021 | 0 / 65,536 |
| `e32,m8` | 454,572 | **272,591** | **852,932** | **1,125,523** | **13,327** | **17,586** | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,028 | 273,927 | 1,545,075 | 1,819,002 | 24,141 | 28,421 | 0 / 65,536 |

Stage fusion improves m2 by `1.30x` and m4 by `1.27x`, but makes m8 `1.81x`
slower. Baseline m8 remains the fastest bin-major kernel. With VLEN=128, the
64-batch row requires eight m2 chunks, four m4 chunks, or two m8 chunks, so
m8's lower chunk count wins until stage fusion creates spill pressure.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 760,957 | **3,995,938** | 7,871,265 | **11,867,203** | 122,988 | **185,425** | 0 / 65,536 |
| `e32,m4` | 766,342 | 4,523,634 | **7,474,891** | 11,998,525 | **116,795** | 187,476 | 0 / 65,536 |
| `e32,m8` | **760,900** | 4,013,433 | 10,979,392 | 14,992,825 | 171,553 | 234,262 | 0 / 65,536 |

Batch-major m2 is its best complete FFT at 11,867,203 cycles, while m4 has
the best batch-major butterfly region. Comparing the best complete kernels,
batch-major is `10.54x` slower than bin-major. Its best butterfly region is
`8.76x` slower. At the same non-fused LMUL, batch-major is `5.79x` slower for
m2, `7.63x` slower for m4, and `13.32x` slower for m8 in complete FFT cycles.

#### Complete LGVV256D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=256, D=128, cold boot,
one shared twiddle plan, and 10 runs per variant. It uses the current
layout-comparison binary. Twiddle plan creation took 191,850 cycles. All nine
variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and use a scalar-broadcast twiddle.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,489 | 205,644 | 1,157,176 | 1,362,820 | 18,080 | 21,294 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,853 | 199,454 | 985,849 | 1,185,303 | 15,403 | 18,520 | 0 / 65,536 |
| `e32,m4` | 453,988 | 219,555 | 970,153 | 1,189,708 | 15,158 | 18,589 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,831 | 198,823 | **847,929** | **1,046,752** | **13,248** | **16,355** | 0 / 65,536 |
| `e32,m8` | 454,726 | **198,192** | 853,838 | 1,052,030 | 13,341 | 16,437 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,256 | 200,140 | 1,214,280 | 1,414,420 | 18,973 | 22,100 | 0 / 65,536 |

Stage fusion improves m2 by `1.17x` and m4 by `1.14x`, while fused m8 is
`1.42x` slower than baseline m8. Fused m4 is the fastest complete bin-major
kernel and is only `1.005x` faster than baseline m8.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT and load vector twiddles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 761,073 | **3,995,939** | 7,261,799 | **11,257,738** | 113,465 | **175,902** | 0 / 65,536 |
| `e32,m4` | 765,540 | 4,524,250 | **7,204,966** | 11,729,216 | **112,577** | 183,269 | 0 / 65,536 |
| `e32,m8` | **760,008** | 4,013,008 | 12,641,780 | 16,654,788 | 197,527 | 260,231 | 0 / 65,536 |

Batch-major m2 is its best complete FFT at 11,257,738 cycles. Comparing the
best complete kernel from each layout, batch-major is `10.75x` slower. Its best
butterfly region, batch-major m4, is `8.50x` slower than bin-major fused m4.
At the same non-fused LMUL, batch-major is `8.26x` slower for m2, `9.86x`
slower for m4, and `15.83x` slower for m8 in complete FFT cycles.

#### Complete LGVV512D128 timing data

This Genesys2 FPGA measurement uses a 50 MHz clock, VLEN=512, D=128, cold boot,
one shared twiddle plan, and 10 runs per LMUL variant. It is the current
layout-comparison binary and explicitly reports `[bin][batch]` as bin-major
and `[batch][bin]` as batch-major. Twiddle plan creation took 191,850 cycles.
All nine variants passed validation with zero mismatches.

##### Bin-major `[bin][batch]`

RVV lanes span the 64 independent batches and each butterfly broadcasts one
scalar twiddle across those lanes.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 464,266 | 158,145 | 861,070 | 1,019,215 | 13,454 | 15,925 | 0 / 65,536 |
| `e32,m2-stage-fused` | 454,708 | 153,892 | 842,949 | 996,841 | 13,171 | 15,575 | 0 / 65,536 |
| `e32,m4` | 453,785 | 173,464 | 828,128 | 1,001,592 | 12,939 | 15,649 | 0 / 65,536 |
| `e32,m4-stage-fused` | 454,716 | 153,707 | **773,128** | **926,835** | **12,080** | **14,481** | 0 / 65,536 |
| `e32,m8` | 454,731 | 152,502 | 854,032 | 1,006,534 | 13,344 | 15,727 | 0 / 65,536 |
| `e32,m8-stage-fused` | 455,373 | 154,892 | 5,316,059 | 5,470,951 | 83,063 | 85,483 | 0 / 65,536 |

Stage fusion improves m2 by `1.02x` and m4 by `1.07x`; fused m4 is the fastest
bin-major kernel. In contrast, fused m8 is `6.22x` slower than baseline m8.
This agrees with the V512D128B disassembly: fused m2 has no whole-register
spill, fused m4 has one `vs4r.v`/`vl4re32.v` pair, and fused m8 contains many
`vs8r.v`/`vl8re32.v` spills.

##### Batch-major `[batch][bin]`

RVV lanes span consecutive bins within one FFT. Consequently each vector
operation loads a vector of twiddles, and bit reversal is performed separately
for all 64 batches.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e32,m2` | 760,478 | **3,995,076** | **7,027,531** | **11,022,607** | **109,805** | **172,228** | 0 / 65,536 |
| `e32,m4` | 765,194 | 4,523,114 | 7,086,306 | 11,609,420 | 110,723 | 181,397 | 0 / 65,536 |
| `e32,m8` | **760,099** | 4,015,303 | 16,912,003 | 20,927,306 | 264,250 | 326,989 | 0 / 65,536 |

Batch-major m2 is its best complete FFT at 11,022,607 cycles. Comparing the
best complete kernel from each layout, batch-major m2 is `11.89x` slower than
bin-major fused m4. Even before bit reversal, its best butterfly region is
`9.09x` slower. At the same non-fused LMUL, batch-major is `10.81x` slower for
m2, `11.59x` slower for m4, and `20.79x` slower for m8 in complete FFT cycles.
The result strongly favors bin-major when 64 independent FFT batches are
available.

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
