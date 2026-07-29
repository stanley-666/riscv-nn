# RVV FFT testbench

完整的 CPU、RVV、Gemmini，FP32/int8，bin-major/batch-major、VLEN/DLEN、
LMUL/fusion 與倍數重算，請見
[`FFT_COMPLETE_RECALCULATED_REPORT.md`](FFT_COMPLETE_RECALCULATED_REPORT.md)。

This testbench implements a single-precision, radix-2 decimation-in-time FFT.
Bit reversal is scalar; each stage uses FP32 LMUL=m4 RVV intrinsics to calculate
groups of complex butterflies in parallel. Input and output complex numbers are
stored in separate real and imaginary arrays.

The input and `numpy.fft.fft` ground truth come from the generated
`fft_vectors.h`. The program reports the maximum real/imaginary component error
and checks every component using

```text
abs(actual - expected) <= 5e-5 + 1e-6 * abs(expected)
```

The relative term accounts for normal FP32 rounding accumulation across FFT
stages while the absolute term still protects expected values near zero. The
program returns a non-zero status if any component exceeds this limit.

## Batched FFT cross-architecture comparison

This section is the single summary for the scalar CPU, RVV, and Gemmini FFT
measurements. Unless noted otherwise, every row processes 64 independent
1,024-point FFTs and reports the complete FFT region: bit reversal plus all
butterfly stages. Input layout, validation, and one-time twiddle-plan creation
are excluded. Lower cycles and time are better.

The current complete discussion of memory rearrangement, VLEN, DLEN,
precision, LMUL, register pressure, fusion, and Gemmini utilization is in
[FFT_COMPLETE_RECALCULATED_REPORT.md](FFT_COMPLETE_RECALCULATED_REPORT.md).
Both bin-major `[bin][batch]` and batch-major `[batch][bin]` are implemented
as RVV kernels and measured on all nine FPGA configurations. Complete tables
are in the [FP32 RVV README](../fft_batched/README.md) and
[Q7 RVV README](../fft_batched_int8/README.md).

CPU and RVV hardware run at 50 MHz. Gemmini hardware runs at 30 MHz, so
wall-clock time, rather than raw cycles, is the fair cross-clock comparison:

$$
t_{\mathrm{ms}}=\frac{\mathrm{cycles}}{f_{\mathrm{MHz}}\times1000}.
$$

### Current hardware results

| Precision | Architecture | Variant/configuration | Clock | FFT cycles | Time for 64 FFTs | Time/FFT | Relative to matching CPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Scalar CPU | baseline | 50 MHz | 12,760,529 | 255.21 ms | 3.988 ms | 1.00x |
| FP32 | RVV | bin-major m4 fused, `LGVV512D128` | 50 MHz | **926,835** | **18.54 ms** | **0.290 ms** | **13.77x faster** |
| FP32 | Scalar CPU | stage-fused | — | pending | pending | pending | — |
| FP32 | Gemmini | baseline/stage-fused | — | not implemented | — | — | — |
| INT8/Q7 | Scalar CPU | baseline | 50 MHz | 13,801,835 | 276.04 ms | 4.313 ms | 1.00x |
| INT8/Q7 | Scalar CPU | stage-fused | 50 MHz | 13,940,562 | 278.81 ms | 4.356 ms | 1.01x slower than CPU baseline |
| INT8/Q7 | RVV | bin-major m4 baseline, `LGVV512D128` | 50 MHz | **1,074,399** | **21.49 ms** | **0.336 ms** | **12.85x faster** |
| INT8/Q7 | Gemmini | current baseline | 30 MHz | 19,073,443 | 635.78 ms | 9.934 ms | **2.30x slower** |
| INT8/Q7 | Gemmini | current stage-fused/native scaling | 30 MHz | pending | pending | pending | — |

The current Gemmini twiddle plan takes 512,544 cycles, or 17.08 ms at 30 MHz,
and is excluded from its FFT row. All populated rows passed their FP32
tolerance or bit-exact INT8 validation.

### Fusion effect on hardware

| Precision | Architecture | Baseline cycles | Fused cycles | Fusion effect |
| --- | --- | ---: | ---: | ---: |
| FP32 | RVV, best-to-best | 987,952 | 926,835 | **6.19% lower latency, 1.07x faster** |
| INT8/Q7 | Scalar CPU | 13,801,835 | 13,940,562 | 1.01% higher latency, 1.01x slower |
| INT8/Q7 | RVV, best-to-best | 1,074,399 | 1,162,509 | 8.20% higher latency, 1.08x slower |
| INT8/Q7 | Gemmini | 19,073,443 | pending | rerun required |

The RVV rows use the fastest complete bin-major FFT for each fusion class.
Batch-major is not pending: its fastest FP32 and Q7 results are 11,022,607
and 7,918,006 cycles. They are not architecture-best rows because they are
11.89x and 7.37x slower than the matching best bin-major kernels.

### Gemmini optimization experiments

The following values are Gemmini Spike results, not 30 MHz FPGA measurements.
They are retained to compare software mappings on the same simulator and do
not replace the current hardware table.

| Gemmini Spike mapping | FFT cycles | Relative to 15,288,837-cycle baseline | Decision |
| --- | ---: | ---: | --- |
| Original WS, Gemmini twiddle + CPU Q7 | **15,288,837** | 1.00x | retained |
| Dense mixed radix 4/16/16 WS | **6,787,804** | 2.26x faster in same-build A/B | retained as hardware-optimized experiment |
| Transposed WS operands | 15,790,053 | 1.03x slower | reverted |
| Stage-level large-buffer WS | 16,166,739 | 1.06x slower | reverted |
| Stage-fused | 17,602,856 | 1.15x slower | not preferred |
| Full Gemmini-native RNE scaling and butterfly matmul | 20,090,711 | 1.31x slower | experimental |

The full native-scaling path passed an independently generated
Gemmini-RNE/saturating reference with 0 / 65,536 mismatches. It is slower
because every butterfly group requires a twiddle matmul followed by a second
butterfly matmul and another DMA/fence boundary. The current radix-2 mapping
uses a sparse 16x16 block-diagonal twiddle matrix with only 32 nonzero
coefficients, or 12.5% arithmetic density.

### Cross-architecture mixed-radix Spike ablation

These values use the same dense mixed-radix 4/16/16 definition, Q7
coefficients, three scaling boundaries, 1,024 points, and 64 batches. They are
Spike results and remain separate from the FPGA radix-2 comparison.

| Architecture | Radix-2 comparison point | Mixed radix 4/16/16 | Effect of mixed radix |
| --- | ---: | ---: | ---: |
| Scalar CPU | 19,377,049 | 56,838,101 | 2.93x slower |
| RVV, VLEN=512 | **322,759** (m4 fused) | 1,715,067 | 5.31x slower |
| Gemmini WS | 15,337,062 | **6,787,804** | **2.26x faster** |

For the same mixed-radix algorithm, RVV uses 3.96x fewer Spike cycles than
Gemmini and 33.14x fewer than the scalar CPU. The ablation shows that dense
radix-16 transforms improve Gemmini utilization, while RVV remains faster
because its radix-2 mapping already exploits the batch dimension without
performing dense direct-DFT MACs.

### Main conclusions

- The fastest measured FP32 batch is RVV bin-major m4 stage fusion at
  926,835 cycles (18.54 ms).
- The fastest measured INT8/Q7 batch is RVV bin-major m4 baseline at
  1,074,399 cycles (21.49 ms).
- Batch-major is implemented and validated on all nine FPGA configurations.
  Its best FP32 and Q7 results are 11,022,607 and 7,918,006 cycles.
- The current Gemmini INT8 baseline takes 635.78 ms: it is 2.30x slower than
  the 50 MHz scalar CPU and 29.59x slower than the best INT8 RVV result in
  wall-clock time.
- Fusion is architecture- and precision-dependent. It helps FP32 RVV but is
  slower for the measured INT8 CPU, RVV, and earlier Gemmini mappings.
- These results show that the sparse radix-2 mapping is inefficient on
  Gemmini. A separate dense mixed-radix 4/16/16 Spike experiment reduces the
  same-build result from 15,337,062 to 6,787,804 cycles. It is not included in
  the controlled radix-2 hardware table until it has been measured on FPGA.
  It also uses three quantization boundaries instead of ten, so it is validated
  against its own mixed-radix Q7 reference rather than claimed to be bit-exact
  with the radix-2 Q7 output.

Detailed measurements and methodology are in the
[FP32 CPU](../fft_cpu/README.md),
[FP32 RVV batch](../fft_batched/README.md),
[INT8 CPU](../fft_cpu_int8/README.md),
[INT8 RVV batch](../fft_batched_int8/README.md), and
[INT8 Gemmini](../fft_batched_int8_gemmini/README.md) READMEs.

Generate vectors for a power-of-two FFT size before building, for example:

```sh
python3 py/fft/fft_groundtruth.py --size 1024 \
    --header testbench/fft/fft_vectors.h
```

Build it for Linux/Spike with an RVV profile:

```sh
./scripts/configure_build.sh linux-pk vector fft zvl128b
```

Example Spike command:

```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft/zvl128b/vector/static/fft
```

### Performance results

All results below use the same 1024-point FP32 LMUL=m4 FFT vectors. Every run
passed with maximum component error `0.000024` at bin 757 and maximum tolerance
ratio `0.385719` at bin 1015.

| Environment | Configuration | Cycles | Relative FPGA cycles | Result |
| --- | --- | ---: | ---: | --- |
| Spike simulation | `zvl128b` | 188,845 | N/A | PASS |
| Genesys2 FPGA | `LGVV256D128` | 255,697 | 1.000x | PASS |
| Genesys2 FPGA | `LGVV512D128` | 255,982 | 1.001x | PASS |
| Genesys2 FPGA | `GENV512D128` | 257,867 | 1.008x | PASS |
| Genesys2 FPGA | `LGVV128D128` | 257,941 | 1.009x | PASS |
| Genesys2 FPGA | `GENV256d128` | 258,187 | 1.010x | PASS |
| Genesys2 FPGA | `GENV128D128` | 260,604 | 1.019x | PASS |
| Genesys2 FPGA | `GENV512D64` | 269,274 | 1.053x | PASS |
| Genesys2 FPGA | `GENV256D64` | 270,163 | 1.057x | PASS |
| Genesys2 FPGA | `GENV128D64` | 271,041 | 1.060x | PASS |

The FPGA relative-cycle column is normalized to the fastest measured hardware
result, `LGVV256D128`. Spike is a simulator, so its cycle count is listed but
is not normalized against the FPGA measurements. These are single reported
runs rather than averages over repeated trials.

Build the same generated vectors as a bare-metal RVV image:

```sh
./scripts/configure_build.sh baremetal vector fft V128D128B
```

The legacy Makefile path is also supported:

```sh
make baremetal \
    TESTBENCH=fft \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

After verifying the target is the intended unmounted whole SD-card device,
flash with:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the intended unmounted whole SD-card device, not a
partition. Flashing overwrites data on the selected device.
