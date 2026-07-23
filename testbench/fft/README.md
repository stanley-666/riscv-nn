# RVV FFT testbench

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

The complete discussion of memory rearrangement, VLEN, DLEN, precision, LMUL,
register pressure, fusion, and Gemmini utilization is available in
[FFT_ARCHITECTURE_ANALYSIS.md](FFT_ARCHITECTURE_ANALYSIS.md).

CPU and RVV hardware run at 50 MHz. Gemmini hardware runs at 30 MHz, so
wall-clock time, rather than raw cycles, is the fair cross-clock comparison:

$$
t_{\mathrm{ms}}=\frac{\mathrm{cycles}}{f_{\mathrm{MHz}}\times1000}.
$$

### Current hardware results

| Precision | Architecture | Variant/configuration | Clock | FFT cycles | Time for 64 FFTs | Time/FFT | Relative to matching CPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | Scalar CPU | baseline | 50 MHz | 12,760,529 | 255.21 ms | 3.988 ms | 1.00x |
| FP32 | RVV | baseline, `LGVV512D128` m8 | 50 MHz | 1,000,357 | 20.01 ms | 0.313 ms | **12.76x faster** |
| FP32 | RVV | stage-fused, `LGVV512D128` m4 | 50 MHz | **926,407** | **18.53 ms** | **0.290 ms** | 13.77x faster than CPU baseline |
| FP32 | Scalar CPU | stage-fused | — | pending | pending | pending | — |
| FP32 | Gemmini | baseline/stage-fused | — | not implemented | — | — | — |
| INT8/Q7 | Scalar CPU | baseline | 50 MHz | 13,801,835 | 276.04 ms | 4.313 ms | 1.00x |
| INT8/Q7 | Scalar CPU | stage-fused | 50 MHz | 13,940,562 | 278.81 ms | 4.356 ms | 1.01x slower than CPU baseline |
| INT8/Q7 | RVV | baseline, `LGVV512D128` m4 | 50 MHz | **1,106,731** | **22.13 ms** | **0.346 ms** | **12.47x faster** |
| INT8/Q7 | RVV | stage-fused, `GENV512D128` m4 | 50 MHz | 1,185,280 | 23.71 ms | 0.370 ms | **11.76x faster than fused CPU** |
| INT8/Q7 | Gemmini | current baseline | 30 MHz | 19,073,443 | 635.78 ms | 9.934 ms | **2.30x slower** |
| INT8/Q7 | Gemmini | current stage-fused/native scaling | 30 MHz | pending | pending | pending | — |

The current Gemmini twiddle plan takes 512,544 cycles, or 17.08 ms at 30 MHz,
and is excluded from its FFT row. All populated rows passed their FP32
tolerance or bit-exact INT8 validation.

### Fusion effect on hardware

| Precision | Architecture | Baseline cycles | Fused cycles | Fusion effect |
| --- | --- | ---: | ---: | ---: |
| FP32 | RVV | 1,000,357 | 926,407 | **7.39% lower latency, 1.08x faster** |
| INT8/Q7 | Scalar CPU | 13,801,835 | 13,940,562 | 1.01% higher latency, 1.01x slower |
| INT8/Q7 | RVV | 1,106,731 | 1,185,280 | 7.10% higher latency, 1.07x slower |
| INT8/Q7 | Gemmini | 19,073,443 | pending | rerun required |

The RVV INT8 baseline and fused rows use the fastest measured configuration
for each variant. If configuration must be held fixed, `LGVV512D128` m4 fused
uses 1,185,495 cycles, only 215 cycles above the listed
`GENV512D128` result.

### Gemmini optimization experiments

The following values are Gemmini Spike results, not 30 MHz FPGA measurements.
They are retained to compare software mappings on the same simulator and do
not replace the current hardware table.

| Gemmini Spike mapping | FFT cycles | Relative to 15,288,837-cycle baseline | Decision |
| --- | ---: | ---: | --- |
| Original WS, Gemmini twiddle + CPU Q7 | **15,288,837** | 1.00x | retained |
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

### Main conclusions

- The fastest measured FP32 batch is RVV m4 stage fusion at 926,407 cycles
  (18.53 ms).
- The fastest measured INT8/Q7 batch is RVV m4 baseline at 1,106,731 cycles
  (22.13 ms).
- The current Gemmini INT8 baseline takes 635.78 ms: it is 2.30x slower than
  the 50 MHz scalar CPU and 28.72x slower than the best INT8 RVV result in
  wall-clock time.
- Fusion is architecture- and precision-dependent. It helps FP32 RVV but is
  slower for the measured INT8 CPU, RVV, and earlier Gemmini mappings.
- These results show that the current sparse radix-2 mapping is inefficient on
  Gemmini. They do not establish the performance of an unimplemented dense
  radix-16 or four-step/matrix-FFT mapping.

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
