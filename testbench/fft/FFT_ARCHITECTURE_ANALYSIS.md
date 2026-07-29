# FFT Architecture Performance Analysis

> **Current data status:** RVV batch-major is implemented, validated, and
> measured on all nine GENV/LGVV FPGA configurations for both FP32 and Q7.
> The canonical recalculation is
> [FFT_COMPLETE_RECALCULATED_REPORT.md](FFT_COMPLETE_RECALCULATED_REPORT.md).
> Older Spike-only layout rows below are ablations, not evidence that the
> FPGA batch-major version is missing.

For the measured cycle and V512D128B disassembly evidence behind the LMUL,
fusion, and spill conclusions, see
[RVV FFT register-pressure analysis](RVV_REGISTER_PRESSURE_ANALYSIS.md).

This document analyzes the existing radix-2 batched FFT on the scalar CPU,
RISC-V Vector Extension (RVV), and Gemmini. It uses the same analysis axes as
the convolution study:

- memory layout and rearrangement;
- VLEN and DLEN;
- numerical precision;
- LMUL;
- register pressure;
- stage fusion;
- accelerator utilization.

## 1. Scope and measurement policy

The workload is fixed by the application:

1. radix-2 FFT;
2. convolution.

The FFT comparison therefore holds radix-2 constant across architectures. It
is a controlled comparison of how well each architecture executes the same
algorithm, not a claim about the best possible FFT algorithm for every type of
hardware.

Unless otherwise stated, every hardware row uses:

- 1,024-point FFTs;
- 64 independent batches;
- complete FFT latency: bit reversal plus all butterfly stages;
- input layout, validation, and one-time twiddle-plan creation excluded;
- 50 MHz for scalar CPU and RVV hardware;
- 30 MHz for Gemmini hardware.

Wall-clock time is calculated as

$$
t_{\mathrm{ms}}
=\frac{\mathrm{cycles}}{f_{\mathrm{MHz}}\times1000}.
$$

Raw cycles must not be used alone to compare the 50 MHz CPU/RVV systems with
the 30 MHz Gemmini system.

## 2. Overall hardware performance

### 2.1 Baseline comparison

| Precision | Architecture | Configuration | FFT cycles | Time/64 FFTs | Time/FFT | Relative to matching CPU |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FP32 | Scalar CPU | baseline | 12,760,529 | 255.21 ms | 3.988 ms | 1.00x |
| FP32 | RVV | `LGVV512D128` bin-major m4 fused | **926,835** | **18.54 ms** | **0.290 ms** | **13.77x faster** |
| FP32 | Gemmini | not implemented | — | — | — | — |
| INT8/Q7 | Scalar CPU | baseline | 13,801,835 | 276.04 ms | 4.313 ms | 1.00x |
| INT8/Q7 | RVV | `LGVV512D128` bin-major m4 baseline | **1,074,399** | **21.49 ms** | **0.336 ms** | **12.85x faster** |
| INT8/Q7 | Gemmini | current WS baseline | 19,073,443 | 635.78 ms | 9.934 ms | **2.30x slower** |

For FP32, RVV reduces complete FFT latency by approximately 92.74% relative
to the scalar CPU. For INT8/Q7, RVV reduces latency by approximately 92.22%.

The current INT8 Gemmini mapping is 2.30x slower than the scalar CPU and
29.59x slower than RVV in wall-clock time. Even before accounting for its
lower clock, Gemmini requires 1.38x as many cycles as the scalar CPU:

$$
\frac{19,073,443}{13,801,835}=1.38.
$$

### 2.2 Fastest measured variants

| Precision | Architecture | Variant | FFT cycles | Time/64 FFTs |
| --- | --- | --- | ---: | ---: |
| FP32 | RVV | `LGVV512D128` bin-major m4 stage-fused | **926,835** | **18.54 ms** |
| INT8/Q7 | RVV | `LGVV512D128` bin-major m4 baseline | **1,074,399** | **21.49 ms** |
| INT8/Q7 | Scalar CPU | baseline | 13,801,835 | 276.04 ms |
| INT8/Q7 | Gemmini | baseline | 19,073,443 | 635.78 ms |

The fastest measured FP32 implementation is RVV m4 stage fusion. The fastest
measured INT8 implementation is RVV m4 without stage fusion.

## 3. Memory layout and rearrangement

Both RVV layouts are hardware implementations, not a bin-major hardware
result compared with a scalar batch-major reference. All nine configurations
run six bin-major baseline/fused variants and three batch-major variants.
Complete tables are in the [FP32 README](../fft_batched/README.md) and
[Q7 README](../fft_batched_int8/README.md).

### 3.1 Scalar CPU layout

The scalar CPU uses a batch-major layout:

$$
[\mathrm{batch}][\mathrm{bin}].
$$

Each scalar FFT traverses 1,024 contiguous bins. This is cache-friendly for a
CPU that completes one FFT at a time, but it does not expose cross-batch SIMD
parallelism.

### 3.2 RVV layout

The preferred RVV mapping uses a bin-major layout:

$$
[\mathrm{bin}][\mathrm{batch}].
$$

The 64 values belonging to one FFT bin are contiguous. A vector butterfly can
therefore process the same bin across many independent FFTs:

```text
bin 0: batch 0 ... batch 63
bin 1: batch 0 ... batch 63
...
```

Bit reversal swaps complete batch rows. The layout conversion has an upfront
cost, but the butterfly region benefits throughout all 10 stages. This mapping
is the main reason radix-2 batched FFT aligns naturally with RVV.

RVV batch-major `[batch][bin]` is also implemented and has FPGA measurements
for every configuration. The global best FP32 batch-major result is
11,022,607 cycles versus 926,835 cycles for bin-major, an **11.89x** penalty.
The global best Q7 batch-major result is 7,918,006 cycles versus 1,074,399
cycles for bin-major, a **7.37x** penalty. Batch-major lanes cover consecutive
bins with different twiddles, early stages expose short vectors, and bit
reversal must be performed independently for every FFT. All variants pass
validation, so this is a measured dataflow/layout effect rather than a
missing implementation or numerical failure.

### 3.3 Gemmini layout

The Gemmini implementation groups eight complex rotations into a 16x16
block-diagonal matrix and packs the 64 batches into a 16x64 input:

$$
Y_{16\times64}=W_{16\times16}X_{16\times64}.
$$

This provides a rectangular DMA source, but every FFT stage selects a
different set of odd bins. The host must repeatedly:

1. gather and pack lower inputs;
2. invoke a weight-stationary Gemmini operation;
3. move out accumulator results;
4. synchronize;
5. complete Q7 processing or invoke a second Gemmini operation.

Convolution pays im2col/packing cost to create a large dense GEMM with strong
weight reuse. The FFT pays rearrangement repeatedly while producing a sparse
GEMM. Memory rearrangement is consequently much more damaging for the current
Gemmini FFT than for Gemmini convolution.

## 4. VLEN analysis

The following tables use the best butterfly kernel for each hardware
configuration. Butterfly cycles exclude bit reversal and isolate the vector
compute region.

### 4.1 FP32, GENV D128

| VLEN | Best kernel | Butterfly cycles | Improvement from previous VLEN |
| ---: | --- | ---: | ---: |
| 128 | m8 | 893,472 | — |
| 256 | m8 | 831,941 | 1.07x |
| 512 | m4 fused | 773,249 | 1.08x |

Increasing VLEN from 128 to 512 improves the best kernel by only 1.16x:

$$
\frac{893,472}{773,249}=1.16.
$$

The scaling is not proportional to VLEN because:

- the batch count is fixed at 64;
- vector instructions eventually cover the entire batch row;
- DLEN limits physical element throughput;
- scalar control and bit reversal do not shrink proportionally;
- larger LMUL and fusion increase register pressure.

### 4.2 INT8/Q7, GENV D128

| VLEN | Best kernel | Butterfly cycles | Improvement from previous VLEN |
| ---: | --- | ---: | ---: |
| 128 | m8 | 1,367,989 | — |
| 256 | m8 | 1,116,785 | 1.22x |
| 512 | m4 | 1,054,211 | 1.06x |

INT8 benefits more when moving from VLEN 128 to 256, but performance again
plateaus from 256 to 512. Once vector coverage is sufficient, increasing VLEN
alone cannot remove widening, rounding, narrowing, and register-lifetime
costs.

## 5. DLEN analysis

### 5.1 FP32

| VLEN | Best D64 cycles | Best D128 cycles | D128 speedup |
| ---: | ---: | ---: | ---: |
| 128 | 1,619,368 | 893,472 | 1.81x |
| 256 | 1,538,435 | 831,941 | 1.85x |
| 512 | 1,479,809 | 773,249 | 1.91x |

### 5.2 INT8/Q7

| VLEN | Best D64 cycles | Best D128 cycles | D128 speedup |
| ---: | ---: | ---: | ---: |
| 128 | 2,243,324 | 1,367,989 | 1.64x |
| 256 | 2,113,540 | 1,116,785 | 1.89x |
| 512 | 2,049,368 | 1,054,211 | 1.94x |

DLEN has a substantially stronger effect than VLEN in these measurements.
VLEN determines how many elements an instruction describes, whereas DLEN more
directly controls how many bits the vector datapath processes per cycle. At
VLEN 512, doubling DLEN from 64 to 128 nearly halves the useful butterfly
cycles for both precisions.

## 6. Precision analysis

### 6.1 Complete FFT latency

| Architecture | FP32 baseline | INT8 baseline | INT8 relative to FP32 |
| --- | ---: | ---: | ---: |
| Scalar CPU | 12,760,529 | 13,801,835 | 1.08x slower |
| RVV | 987,952 | 1,074,399 | 1.09x slower |

INT8 is not faster for this FFT. Unlike a dense INT8 convolution, the
stage-scaled Q7 FFT requires:

- INT8 multiplication with widening;
- INT32 intermediate accumulation;
- Q14-to-Q7 rounding;
- saturation;
- narrowing;
- division by two after every stage.

These operations cost more than the reduced element size saves.

### 6.2 Best butterfly kernel at LGVV512D128

| Precision | Best kernel | Butterfly cycles |
| --- | --- | ---: |
| FP32 | m4 stage-fused | **772,922** |
| INT8/Q7 | m4 baseline | 1,043,142 |

The best INT8 kernel uses 1.35x as many butterfly cycles as the best FP32
kernel:

$$
\frac{1,043,142}{772,922}=1.35.
$$

Gemmini currently has no FP32 FFT implementation, so no within-Gemmini
precision comparison is available.

## 7. LMUL analysis

### 7.1 FP32 at LGVV512D128

| LMUL | Baseline butterfly cycles | Stage-fused butterfly cycles |
| --- | ---: | ---: |
| m2 | 862,190 | 844,371 |
| m4 | 847,393 | **772,922** |
| m8 | 847,515 | 9,403,442 |

The baseline m4 and m8 results are nearly identical. Increasing LMUL no longer
improves throughput after vector coverage is sufficient. Under fusion, m8 is
catastrophically worse because its live vector state exceeds the practical
register budget.

### 7.2 INT8/Q7 at LGVV512D128

| LMUL | Baseline butterfly cycles | Stage-fused butterfly cycles |
| --- | ---: | ---: |
| m2 | 1,124,729 | 1,215,757 |
| m4 | **1,043,142** | 1,121,857 |
| m8 | 1,227,334 | 1,331,394 |

INT8 m8 is already worse without fusion. Complex fixed-point multiplication
widens values and keeps multiple INT32 accumulators alive, so the largest LMUL
reduces the number of independently available vector register groups.

The best LMUL must therefore be selected from the complete dataflow. Choosing
the largest LMUL based only on nominal vector coverage is not reliable.

## 8. Register pressure

The clearest register-pressure result is FP32 m8 stage fusion on
`LGVV512D128`:

$$
\frac{9,403,442}{847,515}=11.10.
$$

Fusion makes m8 11.10x slower than its baseline. Two stages keep more complex
inputs, twiddles, and intermediate results live simultaneously. Whole-register
spill/reload traffic then costs much more than the eliminated stage-boundary
loads and stores.

The effect is precision-dependent:

- FP32 m4 fusion has enough registers to benefit;
- FP32 m8 fusion spills heavily;
- INT8 widening increases live register demand;
- INT8 m8 is unfavorable even before fusion.

This is directly analogous to convolution: more unrolling or accumulator
groups can improve arithmetic coverage until register spills dominate.

## 9. Stage-fusion analysis

| Precision | Architecture | Baseline cycles | Fused cycles | Fusion result |
| --- | --- | ---: | ---: | ---: |
| FP32 | RVV, best-to-best | 987,952 | 926,835 | **1.07x faster** |
| INT8/Q7 | Scalar CPU | 13,801,835 | 13,940,562 | 1.01x slower |
| INT8/Q7 | RVV, best-to-best | 1,074,399 | 1,162,509 | 1.08x slower |
| INT8/Q7 | Gemmini hardware | 19,073,443 | pending | rerun required |

Fusion removes intermediate stage memory traffic, but it also increases:

- live temporary count;
- register pressure;
- spill/reload risk;
- fixed-point conversion complexity;
- Gemmini packing and host/accelerator handoffs.

Fusion is therefore not an architecture-independent optimization. It helps
the FP32 RVV m4 kernel but does not win for the measured INT8 CPU or RVV
kernels.

## 10. Gemmini utilization and experiments

The 16x16 Gemmini twiddle matrix contains eight independent 2x2 complex
rotation blocks:

$$
8\times(2\times2)=32
$$

nonzero coefficients out of 256 positions:

$$
\frac{32}{256}=12.5\%.
$$

The workload has enough points and batches, but increasing the number of
sparse operations does not increase the useful density of each operation.
Gemmini performs dense scheduling around a sparse FFT dependency graph.

### 10.1 Gemmini Spike mapping experiments

| Mapping | FFT cycles | Relative to original WS | Decision |
| --- | ---: | ---: | --- |
| Original WS, Gemmini twiddle + CPU Q7 | **15,288,837** | 1.00x | retained |
| Dense mixed radix 4/16/16 WS | **6,787,804** | 2.26x faster in same-build A/B | retained separately |
| Transposed WS operands | 15,790,053 | 1.03x slower | reverted |
| Stage-level large-buffer WS | 16,166,739 | 1.06x slower | reverted |
| Stage-fused | 17,602,856 | 1.15x slower | not preferred |
| Gemmini-native scaling + butterfly matmul | 20,090,711 | 1.31x slower | experimental |

The `transpose_A` WS mapping was also measured as a same-build A/B test.
It kept the input in its existing physical layout, used Gemmini's transpose
support, and pretransposed the twiddle weights:

| Same-build WS variant | FFT cycles | Difference |
| --- | ---: | ---: |
| Baseline | **15,334,869** | reference |
| Gemmini `transpose_A` + pretransposed weights | 15,336,071 | +1,202 (+0.008%) |

Both variants produced 0 / 65,536 mismatches. Because the transposed variant
was not faster, it was reverted rather than replacing the retained baseline.
These same-build values should not be mixed with the earlier 15,288,837-cycle
measurement when calculating speedup.

The dense mixed-radix implementation uses the factorization
$1024=4\times16\times16$. The execution order 4/16/16 minimizes the number of
distinct dense transforms: one radix-4 transform, four radix-16 transforms,
and 64 radix-16 transforms, for 69 Gemmini calls per FFT. Each call combines
all applicable FFT blocks and all 64 batches. The real representation is an
8x8 or 32x32 dense matrix rather than the radix-2 16x16 block-diagonal matrix.

In the same build, the result was:

| Variant | FFT cycles | Relative result |
| --- | ---: | ---: |
| Radix-2 WS baseline | 15,337,062 | reference |
| Dense mixed radix 4/16/16 WS | **6,787,804** | **2.26x faster; 55.7% fewer cycles** |

The accelerator output has 0 / 65,536 mismatches against an independent
scalar mixed-radix Q7 reference. It is not bit-exact with the existing
radix-2 Q7 reference because the mixed-radix algorithm rounds and scales at
three stage boundaries instead of ten. Against the radix-2 output, 60,992 /
65,536 complex points have both components within +/-1, the mean absolute
component difference is 0.533, and the maximum component difference is 3.
This result therefore belongs to a hardware-optimized algorithm comparison,
not the controlled radix-2 comparison. FPGA timing is still required.

The same mixed-radix definition was also implemented on the scalar CPU and
RVV and simulated with Spike:

| Architecture | Radix-2 comparison point | Mixed radix 4/16/16 | Radix change |
| --- | ---: | ---: | ---: |
| Scalar CPU | 19,377,049 | 56,838,101 | 2.93x slower |
| RVV, VLEN=512 | **322,759** (m4 fused) | 1,715,067 | 5.31x slower |
| Gemmini WS | 15,337,062 | **6,787,804** | **2.26x faster** |

All three mixed-radix implementations match the shared scalar mixed-radix
reference at all 65,536 complex points. For the same mixed-radix workload,
RVV requires 3.96x fewer Spike cycles than Gemmini. The opposite effect of the
radix change is architectural: direct dense transforms expose Gemmini's
systolic array but add unnecessary MACs for the CPU and RVV. RVV radix-2
already applies one twiddle scalar to a contiguous 64-batch vector and skips
the zeros implicit in the stage matrix.

Gemmini-native output scaling, saturation, and butterfly add/subtract were
functionally validated against an independent RNE reference with
0 / 65,536 mismatches. They are slower because the second butterfly matmul
adds another packing, DMA, execution, output, and fence boundary.

### 10.2 Interpretation

The current result does not show that all systolic FFT architectures are
inefficient. Published FFT-specific systolic architectures use:

- collapsed butterfly cells;
- FIFO/delay networks;
- stage-to-stage streaming;
- resident twiddles;
- continuous pipelining without a memory round trip after each stage.

Gemmini is instead a dense-GEMM systolic array. The current software maps
radix-2 FFT to sparse matrix operations and returns to memory/host processing
between operations. The correct conclusion is limited to this mapping:

> Gemmini's dense-GEMM systolic array is poorly matched to the current sparse
> radix-2 FFT formulation. This does not establish the performance limit of an
> FFT-specific systolic pipeline or an unimplemented dense blocked FFT.

## 11. Relationship to convolution

| Analysis factor | Convolution | Radix-2 FFT |
| --- | --- | --- |
| Memory rearrangement | im2col and weight packing | bin/batch layout, bit reversal, Gemmini packing |
| VLEN | channel/output coverage | batch-row coverage |
| DLEN | MAC throughput | butterfly throughput |
| Precision | INT8 dense-MAC advantage | widening, rounding, saturation overhead |
| LMUL | accumulator and unroll groups | complex temporaries and batch vectors |
| Register pressure | channel accumulation/unrolling | stage fusion and widening |
| Accelerator utilization | dense GEMM and weight reuse | sparse block-diagonal GEMM |
| Preferred hardware | Gemmini can be favorable | RVV is favorable in current measurements |

The two operators expose complementary architectural strengths:

- RVV maps naturally to batch-parallel radix-2 butterflies;
- Gemmini maps naturally to dense convolution with high weight reuse.

For a heterogeneous SoC containing both units, a promising end-to-end mapping
is RVV for FFT and Gemmini for convolution, including the layout-conversion
cost between them in the total latency.

## 12. Conclusions

1. RVV is the fastest measured architecture for both FP32 and INT8 radix-2
   batched FFT.
2. DLEN has a larger effect than VLEN once the 64-batch row has sufficient
   vector coverage.
3. INT8 is slower than FP32 on CPU and RVV because fixed-point widening,
   rounding, saturation, and narrowing dominate the data-size benefit.
4. m4 is the most robust LMUL. m8 can suffer severe register spilling,
   especially under fusion.
5. Fusion helps FP32 RVV m4 but is not beneficial for the measured INT8 CPU,
   RVV, or earlier Gemmini paths.
6. The current Gemmini radix-2 mapping is limited by sparse utilization,
   packing, DMA, output, synchronization, and host/accelerator boundaries.
7. Fixing radix-2 is fair because radix-2 is part of the application workload.
   The conclusions must remain scoped to radix-2 rather than generalized to
   every possible FFT architecture.

## 13. Data sources

- [Consolidated FFT comparison](README.md#batched-fft-cross-architecture-comparison)
- [FP32 scalar CPU](../fft_cpu/README.md)
- [FP32 RVV batch](../fft_batched/README.md)
- [INT8 scalar CPU](../fft_cpu_int8/README.md)
- [INT8 RVV batch](../fft_batched_int8/README.md)
- [INT8 Gemmini](../fft_batched_int8_gemmini/README.md)

Relevant systolic FFT literature:

- Valentin Boriakoff, “FFT Computation with Systolic Arrays, A New
  Architecture,” IEEE Transactions on Circuits and Systems II, 1994.
- Preston A. Jackson et al., “A Systolic FFT Architecture for Real Time FPGA
  Systems,” HPEC, 2004.
