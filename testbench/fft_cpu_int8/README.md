<!-- SPDX-FileContributor: Person: Stanley Lee -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Pure CPU scaled int8 FFT

Linux/Spike and bare-metal compile the same
`testbench/fft_cpu_int8/fft_cpu_int8.c` source and execute the same `main()`.
Bare-metal replaces only startup, linking, minilib, and the `nn_runtime`
implementation; there is no separate application adapter.
The bare-metal startup prints hardware information before entering this shared
`main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

This is the pure CPU baseline for `fft_batched_int8`, with RVV disabled and
compiler auto-vectorization disabled. It processes 64 independent
1024-point complex FFTs in contiguous `[batch][bin]` layout and uses the same
Q7 twiddles, int32 complex accumulators, RNU rounding, int8 saturation, and
per-stage division by two. The total transform scale is $1/1024$.

The input and bit-exact ground truth are generated with:

```sh
python3 py/fft/fft_int8_groundtruth.py --size 1024 \
    --header testbench/fft_cpu_int8/fft_int8_vectors.h
```

## Linux/pk and Spike

```sh
./scripts/configure_build.sh linux-pk cpu fft_cpu_int8 default

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d pk \
    build/linux-pk/fft_cpu_int8/default/cpu/static/fft_cpu_int8
```

Compiler auto-vectorization is disabled, and disassembly of
`fft_cpu_int8.c.o` contains no vector instructions, so the measured FFT
kernels remain scalar. The current statically linked runtime/libc contains RVV
instructions and therefore requires V in Spike's ISA string even though the
testbench object and measured scalar kernels do not use RVV.
The same binary runs both the original stage-by-stage scalar kernel and a
two-stage-fused scalar kernel. Both share one twiddle plan and run 10 times.
The fused kernel preserves the first stage's RNU scaling, saturation, and Q7
quantization in scalar temporaries before evaluating the second stage, so
fusion removes only the intermediate memory store/reload.

## Bare-metal build

```sh
./scripts/configure_build.sh baremetal cpu fft_cpu_int8 cpu
```

The output files are:

```text
build/baremetal/fft_cpu_int8/cpu_nn_cpu_baremetal.elf
build/baremetal/fft_cpu_int8/cpu_nn_cpu_baremetal.bin
```

## Bare-metal flash

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft_cpu_int8 \
    BACKEND=cpu \
    HARDWARE_CONFIG=cpu \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. Flashing
overwrites data on the selected device.

For both `baseline` and `stage-fused`, the testbench reports 10-run average
input-layout, bit-reversal, butterfly, FFT, and reused-plan cycles, plus
first-run-equivalent cycles. Validation checks all 65,536 complex points for
exact equality and prints the first 64 mismatches.

## Performance results

### Current baseline and stage-fused Spike result

Both variants passed bit-exact validation with `0 / 65,536` mismatches.
Twiddle-plan creation took 115,984 cycles.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Butterfly cycles/FFT | FFT cycles/FFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 661,411 | 3,105,992 | 14,176,852 | 17,282,844 | 221,513 | 270,044 |
| stage-fused | 590,278 | 3,105,991 | 12,892,427 | 15,998,418 | 201,444 | 249,975 |

Scalar stage fusion improves the butterfly region by `1.10x`. For a fair RVV
comparison, baseline CPU should be compared with baseline RVV, and fused CPU
with fused RVV.

### Dense mixed-radix Spike ablation

The CPU binary also runs the same dense mixed-radix 4/16/16 algorithm used by
the RVV and Gemmini experiments. This is a hardware-aware ablation, not a
replacement for the controlled radix-2 result.

| Scalar CPU Spike variant | FFT cycles | Relative to radix-2 |
| --- | ---: | ---: |
| Radix-2 baseline | **19,377,049** | 1.00x |
| Dense mixed radix 4/16/16 | 56,838,101 | **2.93x slower** |

The mixed-radix result has 0 / 65,536 mismatches against the shared scalar
mixed-radix reference. Relative to the radix-2 Q7 output, 60,992 / 65,536
complex points have both components within +/-1, the mean absolute component
difference is 0.533, and the maximum component difference is 3. Direct dense
radix transforms increase scalar operation count, so fewer stages do not
compensate for the additional MACs on the CPU.

### Current 50 MHz Genesys2 pure-CPU result

This cold-cache bare-metal result uses the same 10-run averaging and bit-exact
validation policy. Both variants passed with `0 / 65,536` mismatches.

| Variant | Input layout | Bit reversal | Butterfly | FFT | Reused plan | First-run equivalent | Butterfly cycles/FFT | FFT cycles/FFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 616,591 | 2,340,831 | **11,461,004** | **13,801,835** | **14,418,426** | **14,605,811** | **179,078** | **215,653** |
| stage-fused | 541,599 | 2,302,478 | 11,638,084 | 13,940,562 | 14,482,161 | 14,669,546 | 181,845 | 217,821 |

Twiddle-plan creation took 187,385 cycles. Unlike Spike, scalar stage fusion
is `1.02x` slower in the butterfly region on this hardware. Its lower input
layout and bit-reversal averages reduce the overall penalty, but the
reused-plan path is still about 0.44% slower than baseline. The pure-CPU
hardware reference should therefore use the baseline result.

### Archived pre-fusion results

The following rows are single cold-start runs of the earlier pure CPU
implementation rather than multi-run averages:

> These rows predate the hot-loop branch hoist and separate bit-reversal /
> butterfly counters. Re-run the current binary before using them as the final
> scalar baseline.

| Environment | FFT size | Batches | Scale | Twiddle plan cycles | Input layout cycles | FFT cycles | Reused-plan total cycles | First-run total cycles | FFT cycles/FFT | Reused-plan cycles/FFT | First-run cycles/FFT | Mismatches | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation, RV64GC pure CPU (RVV disabled) | 1,024 | 64 | $1/1024$ | 111,272 | 651,367 | 20,973,008 | 21,624,375 | 21,735,647 | 327,703 | 337,880 | 339,619 | 0 / 65,536 | PASS |
| Genesys2 FPGA, `GENV128D64`, pure CPU (RVV disabled), 50 MHz, cold cache | 1,024 | 64 | $1/1024$ | 193,105 | 555,448 | 13,991,874 | 14,547,322 | 14,740,427 | 218,623 | 227,301 | 230,319 | 0 / 65,536 | PASS |

The maximum component error is zero under the bit-exact Q7 definition.
