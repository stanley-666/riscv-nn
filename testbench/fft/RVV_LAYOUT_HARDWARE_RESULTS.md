# FFT memory-layout and RVV FPGA results

This document summarizes the current 1,024-point, 64-batch Genesys2 FPGA
layout campaign. CPU and RVV results use 50 MHz. Every listed RVV result passed
validation with 0 / 65,536 mismatches.

## Implemented layouts

| Architecture/kernel | Physical layout | Vector lanes span | Hardware status |
| --- | --- | --- | --- |
| Scalar CPU | `[batch][bin]` | scalar | measured |
| RVV bin-major | `[bin][batch]` | independent batches at one bin | measured on all 9 configurations |
| RVV batch-major | `[batch][bin]` | consecutive bins in one FFT | measured on all 9 configurations |

RVV batch-major is an actual FPGA kernel, not a scalar reference and not only
a Spike ablation. Each FP32 configuration runs six bin-major baseline/fused
variants plus three batch-major variants. Each Q7 configuration additionally
runs the dense mixed-radix experiment.

At 50 MHz:

$$
t_{\mathrm{ms}}=\frac{\mathrm{cycles}}{50{,}000}.
$$

`FFT cycles` includes bit reversal and butterfly processing, but excludes
input-layout conversion and one-time plan construction.

## Best layout result per configuration

### FP32

| Configuration | Best bin-major | Variant | Best batch-major | Variant | Batch penalty |
| --- | ---: | --- | ---: | --- | ---: |
| GENV128D64 | 1,933,799 | m4 fused | 12,837,967 | m2 | 6.64x |
| GENV128D128 | 1,169,218 | m8 | 12,368,468 | m4 | 10.58x |
| GENV256D64 | 1,774,230 | m4 fused | 12,238,898 | m2 | 6.90x |
| GENV256D128 | 1,033,571 | m8 | 11,686,343 | m2 | 11.31x |
| GENV512D64 | 1,687,393 | m4 fused | 12,107,773 | m2 | 7.18x |
| GENV512D128 | 927,136 | m4 fused | 11,435,252 | m2 | 12.33x |
| LGVV128D128 | 1,125,523 | m8 | 11,867,203 | m2 | 10.54x |
| LGVV256D128 | 1,046,752 | m4 fused | 11,257,738 | m2 | 10.75x |
| LGVV512D128 | **926,835** | m4 fused | **11,022,607** | m2 | 11.89x |

### INT8/Q7

| Configuration | Best bin-major | Variant | Best batch-major | Variant | Batch penalty |
| --- | ---: | --- | ---: | --- | ---: |
| GENV128D64 | 2,320,895 | m8 | 9,350,701 | m4 | 4.03x |
| GENV128D128 | 1,418,607 | m8 | 8,485,372 | m4 | 5.98x |
| GENV256D64 | 2,151,407 | m4 | 9,112,563 | m4 | 4.24x |
| GENV256D128 | 1,178,280 | m8 | 8,219,844 | m4 | 6.98x |
| GENV512D64 | 2,097,075 | m4 | 9,039,170 | m4 | 4.31x |
| GENV512D128 | 1,079,519 | m4 | 8,129,785 | m4 | 7.53x |
| LGVV128D128 | 1,374,580 | m4 | 8,215,474 | m4 | 5.98x |
| LGVV256D128 | 1,150,522 | m4 | 8,008,818 | m4 | 6.96x |
| LGVV512D128 | **1,074,399** | m4 | **7,918,006** | m4 | 7.37x |

## Why batch-major is slower

The result does not mean batch-major is missing or scalar:

1. Bin-major makes the 64 independent batches contiguous, so one scalar
   twiddle is broadcast across vector lanes.
2. Batch-major lanes cover different bins, so lanes need different twiddles
   and early radix-2 stages expose short useful vectors.
3. Batch-major bit reversal is repeated for every FFT. Bin-major swaps two
   contiguous 64-element batch rows.
4. Larger LMUL cannot remove this dataflow problem and can add register-group
   pressure; m8 is not the best batch-major choice in the FPGA results.

Consequently, the global FP32 batch-major result is 11.89x slower than the
global FP32 bin-major result. For Q7 the penalty is 7.37x. The smaller Q7
penalty comes partly from packing more e8 bin lanes, but Q7 batch-major is
still much slower than Q7 bin-major.

## Detailed sources

- [Complete recalculated report](FFT_COMPLETE_RECALCULATED_REPORT.md)
- [FP32 per-variant FPGA counters](../fft_batched/README.md)
- [Q7 per-variant FPGA counters](../fft_batched_int8/README.md)
- [Register-pressure and disassembly analysis](RVV_REGISTER_PRESSURE_ANALYSIS.md)
