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

The batch-major input `[batch][bin]` is transposed in a measured preprocessing
step into `[bin][batch]`. Each radix-2 butterfly uses scalar real/imaginary
twiddle factors and LMUL=m8 vectors spanning independent batches. The
butterfly is isolated in `butterfly_rvv_f32()` and written entirely with RVV
intrinsics. Twiddle-factor generation is isolated in `twiddle_f32()` so the
stage loop only describes index pairing and butterfly execution. Compiler-only
fences preserve the intended vector-load to
floating-point element-group chaining order without emitting target
instructions.

The stage loop computes each `(stage, offset)` twiddle only once and reuses it
across all blocks. The `offset == 0` case uses `butterfly_unity_rvv_f32()`,
which replaces multiplication by $1+0j$ with vector add/sub operations.

The input and working arrays are explicitly aligned to 64-byte boundaries.
Memory reordering is measured separately from FFT execution.

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

The recorded results below use the earlier LMUL=m4 butterfly kernel. New
LMUL=m8 measurements should be recorded separately rather than compared as if
they came from the current binary.

Each row below is a single Spike run with 64 identical 1024-point FP32 FFTs.
The two earlier m4 runs passed with maximum component error `0.000024` at
batch 0 bin 757 and maximum tolerance ratio `0.385719` at batch 0 bin 1015.

| Environment | Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, LMUL=m4 | 64 | 827,078 | 1,104,678 | 1,931,756 | 17,260 | 30,183 | PASS |
| Spike simulation | `zvl512b`, LMUL=m4 | 64 | 827,078 | 608,994 | 1,436,072 | 9,515 | 22,438 | PASS |
| Spike simulation | `zvl128b`, LMUL=m8 | 64 | 827,076 | 665,325 | 1,492,401 | 10,395 | 23,318 | PASS |

The `zvl128b`, LMUL=m8 run used the current kernel and passed all 65,536
complex points with zero non-finite outputs. Its maximum component error was
`0.000026` and maximum tolerance ratio was `0.461186`, both at batch 0,
bin 1015.

Input initialization, validation, and bin printing are outside the measured
regions. These values are from one run rather than an average.

### Genesys2 FPGA configuration matrix

The bit-reversal swap is currently restricted to `e32m1` strip mining as a
hardware-isolation workaround. The current butterfly arithmetic uses `e32m8`.
This avoids the compiler-generated `e8,m1` plus 32-bit vector load/store
sequence (effective EMUL=m4) while testing the arithmetic path at m8.

With this workaround, the following Genesys2 runs pass all 65,536 complex
points. These are single cold-cache runs, not warm-cache measurements or
multi-run averages. The `LGVV512D128` row is from the build timestamped
`20260722 114201`:

| Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Max component error | Max tolerance ratio | Mismatches | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `V512D128` (bit reversal `e32m1`, butterfly `e32m4`) | 64 | 3,190,883 | 1,373,879 | 4,564,762 | 21,466 | 71,324 | 0.000024 | 0.385719 | 0 / 65,536 | PASS |
| `GENV128D64` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,241,677 | 2,196,989 | 5,438,666 | 34,327 | 84,979 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `GENV128D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,191,320 | 1,458,026 | 4,649,346 | 22,781 | 72,646 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `GENV256D64` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,242,023 | 2,004,708 | 5,246,731 | 31,323 | 81,980 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `GENV256D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,195,854 | 1,256,358 | 4,452,212 | 19,630 | 69,565 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `GENVV512D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,190,110 | 1,179,164 | 4,369,274 | 18,424 | 68,269 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `LGVV128D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,192,459 | 1,422,060 | 4,614,519 | 22,219 | 72,101 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `LGVV256D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,194,513 | 1,198,946 | 4,393,459 | 18,733 | 68,647 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |
| `LGVV512D128` (bit reversal `e32m1`, butterfly `e32m8`) | 64 | 3,192,232 | 1,119,913 | 4,312,145 | 17,498 | 67,377 | 0.000026 | 0.461186 | 0 / 65,536 | PASS |

### Speedup over scalar CPU

The speedup baseline is the 50 MHz Genesys2 scalar CPU cold-cache result from
`testbench/fft_cpu`: 60,753,733 cycles for 64 FFTs, or 949,277 cycles per FFT.
All compared CPU and RVV runs passed with zero mismatches out of 65,536 complex
points. Kernel speedup excludes the RVV-only matrix reorder; end-to-end speedup
uses `total cycles/FFT` and therefore includes that reorder cost.

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
S_{\mathrm{layout+FFT}}
=\frac{C_{\mathrm{CPU/FFT}}}{C_{\mathrm{RVV\ total/FFT}}}.
$$

| RVV configuration | RVV FFT cycles/FFT | RVV total cycles/FFT | Kernel speedup vs CPU | End-to-end speedup vs CPU |
| --- | ---: | ---: | ---: | ---: |
| `GENV128D64` | 34,327 | 84,979 | 27.65x | 11.17x |
| `GENV128D128` | 22,781 | 72,646 | 41.67x | 13.07x |
| `GENV256D64` | 31,323 | 81,980 | 30.31x | 11.58x |
| `GENV256D128` | 19,630 | 69,565 | 48.36x | 13.65x |
| `GENVV512D128` | 18,424 | 68,269 | 51.52x | 13.90x |
| `LGVV128D128` | 22,219 | 72,101 | 42.72x | 13.17x |
| `LGVV256D128` | 18,733 | 68,647 | 50.67x | 13.83x |
| `LGVV512D128` | 17,498 | 67,377 | 54.25x | 14.09x |

The passing m4 and m8 butterflies rule out a general high-LMUL floating-point
pipeline failure. The `e32m1` bit-reversal form is retained for FPGA
compatibility and deterministic validation.
