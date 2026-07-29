# FFT memory-layout and RVV FPGA results

This document consolidates the current Genesys2 FPGA measurements for the
1,024-point, 64-batch FFT. All systems run at 50 MHz and all listed variants
passed validation.

| Architecture | Physical layout | Contiguous dimension | Parallelization |
| --- | --- | --- | --- |
| Scalar CPU | batch-major `[batch][bin]` | bins of one FFT | one batch at a time |
| RVV | bin-major `[bin][batch]` | 64 batches of one bin | one butterfly across batches |

There is no RVV batch-major FPGA measurement. A separate Spike layout
ablation now compares RVV bin-major and batch-major kernels, but it must not be
mixed into the FPGA tables below.

For FP32, m2/m4/m8 denotes the e32 vector LMUL. For INT8/Q7, it denotes the
LMUL of the widening INT32 accumulator; the corresponding INT8 source groups
are smaller so that widening preserves the same lane count.

At 50 MHz:

$$
t_{\mathrm{ms}}=\frac{\mathrm{cycles}}{50{,}000}.
$$

## 1. Batch-major scalar CPU

| Precision | Variant | Input layout | Bit reversal | Butterfly | FFT cycles | FFT ms | Reused-plan total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | baseline | 759,907 | included in FFT | — | 12,760,529 | 255.21 | 13,520,436 |
| INT8/Q7 | baseline | 616,591 | 2,340,831 | 11,461,004 | 13,801,835 | 276.04 | 14,418,426 |
| INT8/Q7 | stage-fused | 541,599 | 2,302,478 | 11,638,084 | 13,940,562 | 278.81 | 14,482,161 |

The CPU has no LMUL choice. Batch-major avoids a transpose but does not expose
the 64 batches to one vector instruction.

## 2. RVV INT8 layout ablation on Spike

Both layouts use the same radix-2 Q7 definition, VLEN=512, 64 batches, and
10-run averages. All variants produced 0 / 65,536 mismatches.

| LMUL | Bin-major FFT cycles | Batch-major FFT cycles | Batch-major penalty |
| --- | ---: | ---: | ---: |
| m2 | 621,666 | 5,080,368 | 8.17x |
| m4 | **386,632** | 4,963,632 | 12.84x |
| m8 | 407,111 | **4,919,856** | 12.08x |

| Best-kernel counter | Bin-major m4 | Batch-major m8 | Batch-major penalty |
| --- | ---: | ---: | ---: |
| Input layout | 665,608 | 917,960 | 1.38x |
| Bit reversal | 94,792 | 1,065,673 | 11.24x |
| Butterfly | 291,840 | 3,854,183 | 13.21x |
| Complete FFT | **386,632** | **4,919,856** | **12.72x** |

Bin-major vectorizes one butterfly across 64 contiguous batches and swaps an
entire batch row during bit reversal. Batch-major vectorizes offsets within
one FFT: twiddles become vector-vector operands, early stages have short
vectors, and bit reversal uses precomputed indices with an out-of-place
gather/copy for each of the 64 FFTs.

## 3. Bin-major RVV: FP32 non-fused complete FFT

Each cell is `complete FFT cycles (ms)`. FFT includes bit reversal and
butterflies but excludes input layout and one-time plan creation.

| Hardware | VLEN | DLEN | m2 | m4 | m8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GENV128D64` | 128 | 64 | 2,565,325 (51.31) | 2,068,133 (41.36) | **1,970,658 (39.41)** |
| `GENV128D128` | 128 | 128 | 2,161,731 (43.23) | 1,450,100 (29.00) | **1,165,806 (23.32)** |
| `GENV256D64` | 256 | 64 | 2,007,981 (40.16) | **1,775,314 (35.51)** | 1,910,686 (38.21) |
| `GENV256D128` | 256 | 128 | 1,482,821 (29.66) | 1,101,188 (22.02) | **1,030,233 (20.60)** |
| `GENV512D64` | 512 | 64 | **1,881,665 (37.63)** | 1,887,747 (37.75) | 1,882,702 (37.65) |
| `GENV512D128` | 512 | 128 | 1,067,091 (21.34) | **981,232 (19.62)** | 984,229 (19.68) |
| `LGVV128D128` | 128 | 128 | 2,050,547 (41.01) | 1,391,206 (27.82) | **1,125,754 (22.52)** |
| `LGVV256D128` | 256 | 128 | 1,362,474 (27.25) | 1,050,042 (21.00) | **1,045,058 (20.90)** |
| `LGVV512D128` | 512 | 128 | 1,020,211 (20.40) | 1,001,511 (20.03) | **1,000,357 (20.01)** |

Best non-fused: `GENV512D128` m4, 981,232 cycles. With fusion, the global
best is `LGVV512D128` m4 fused at 926,407 cycles, or 18.53 ms.

## 4. Bin-major RVV: FP32 butterfly and fusion

Values exclude input layout and bit reversal.

| Hardware | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best |
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

## 5. Bin-major RVV: INT8/Q7 non-fused complete FFT

Each cell is `complete FFT cycles (ms)`.

| Hardware | VLEN | DLEN | m2 | m4 | m8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GENV128D64` | 128 | 64 | 3,338,255 (66.77) | 2,570,466 (51.41) | **2,362,959 (47.26)** |
| `GENV128D128` | 128 | 128 | 2,591,594 (51.83) | 1,832,321 (36.65) | **1,479,120 (29.58)** |
| `GENV256D64` | 256 | 64 | 2,554,850 (51.10) | **2,213,203 (44.26)** | 2,293,132 (45.86) |
| `GENV256D128` | 256 | 128 | 1,747,493 (34.95) | 1,353,794 (27.08) | **1,208,076 (24.16)** |
| `GENV512D64` | 512 | 64 | 2,177,939 (43.56) | **2,114,418 (42.29)** | 2,389,651 (47.79) |
| `GENV512D128` | 512 | 128 | 1,330,588 (26.61) | **1,117,864 (22.36)** | 1,291,060 (25.82) |
| `LGVV128D128` | 128 | 128 | 2,162,411 (43.25) | 1,476,907 (29.54) | **1,433,119 (28.66)** |
| `LGVV256D128` | 256 | 128 | 1,401,623 (28.03) | **1,207,711 (24.15)** | 1,207,983 (24.16) |
| `LGVV512D128` | 512 | 128 | 1,188,439 (23.77) | **1,106,731 (22.13)** | 1,290,920 (25.82) |

Best INT8: `LGVV512D128` m4 at 1,106,731 cycles, or 22.13 ms.

## 6. Bin-major RVV: INT8/Q7 butterfly and fusion

Values exclude input layout and bit reversal.

| Hardware | m2 | m2 fused | m4 | m4 fused | m8 | m8 fused | Best |
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

## 7. Layout overhead and conclusions

| Precision | Typical RVV bin-major input-layout cycles | Approximate time |
| --- | ---: | ---: |
| FP32 | 444,000 to 465,000 | 8.88 to 9.30 ms |
| INT8/Q7 | 1,140,000 to 1,143,000 | 22.80 to 22.86 ms |

- RVV's bin-major layout enables one twiddle scalar to operate on the
  contiguous 64-batch vector.
- Increasing LMUL is not monotonically beneficial. VLEN=128/DLEN=128 usually
  favors m8; VLEN=512/DLEN=128 generally favors m4.
- DLEN=128 often matters more than increasing VLEN after all 64 batches are
  already covered.
- FP32 fused m8 at VLEN=512 is approximately 11x slower because of register
  spills. INT8 fusion also does not beat the best non-fused hardware kernel.
- Papers must state whether bin-major input-layout conversion is excluded,
  amortized, or included in end-to-end latency.

Detailed per-configuration counters remain in:

- [FP32 RVV results](../fft_batched/README.md)
- [INT8 RVV results](../fft_batched_int8/README.md)
- [FP32 CPU results](../fft_cpu/README.md)
- [INT8 CPU results](../fft_cpu_int8/README.md)
