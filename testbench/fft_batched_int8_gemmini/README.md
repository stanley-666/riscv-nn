# Batched scaled INT8 FFT on Gemmini

This testbench evaluates the same 1,024-point, 64-batch, stage-scaled Q7 FFT
used by `fft_cpu_int8` and `fft_batched_int8`.

Each Gemmini operation combines eight non-unity butterflies. Their complex
twiddle rotations are packed into a 16x16 block-diagonal matrix and multiplied
by a 16x64 lower-input matrix using Gemmini's weight-stationary dataflow.
Gemmini returns int32 accumulators. The host then applies the same RNU
Q14-to-Q7 conversion, saturation, upper/lower add-subtract, and stage division
by two as the CPU and RVV implementations. Unity butterflies remain on the
host because Q7 `127` is not an exact representation of 1.0.

The `wi=-128` case would require an unrepresentable `+128` matrix coefficient.
The testbench uses `+127` and adds the missing odd-imaginary term to the int32
accumulator before rounding, preserving bit-exact arithmetic.

The twiddle table and all 513 Gemmini twiddle tiles are created once. Every
tile permanently stores its 16x16 block-diagonal weight matrix, even/odd bin
indices, active-lane count, and INT8 correction metadata. The FFT runs 10
times, reloading the original input before each run, and reports the average
FFT cycles. Validation checks all 65,536 complex outputs against
`fft_int8_vectors.h` and prints the first 64 mismatches.

The same binary also measures a two-stage-fused implementation. Four
independent four-bin groups are packed per fused tile. The first stage performs
up to eight Gemmini complex rotations; the host applies the mandatory Q7
rounding, saturation, and `1/2` scaling into a temporary tile buffer. Those
quantized values are packed directly into the second Gemmini operation, and
only the second-stage outputs are written to the global FFT data buffers. The
fused path therefore removes the global intermediate stage store/reload but
does not remove either stage's numerical conversion.

## Mapping radix-2 FFT to Gemmini

The implementation remains a radix-2 decimation-in-time FFT with
$\log_2(1024)=10$ stages. It does not replace FFT with a dense $1024\times1024$
DFT matrix. Input is stored as contiguous `[bin][batch]` rows, and the host
performs bit reversal once at the beginning of each FFT run.

For an upper complex input $a=a_r+j a_i$, lower input $b=b_r+j b_i$, and
twiddle $w=w_r+j w_i$, the rotation in one butterfly is

$$
\begin{aligned}
t_r &= w_r b_r-w_i b_i,\\
t_i &= w_i b_r+w_r b_i.
\end{aligned}
$$

This rotation is a $2\times2$ matrix multiplication:

$$
\begin{bmatrix}t_r\\t_i\end{bmatrix}
=
\begin{bmatrix}w_r&-w_i\\w_i&w_r\end{bmatrix}
\begin{bmatrix}b_r\\b_i\end{bmatrix}.
$$

Calling Gemmini for one $2\times2$ multiplication would spend most of the time
on command and data-transfer overhead. The testbench therefore groups eight
independent non-unity butterflies from the same FFT stage. Their eight twiddle
matrices form one $16\times16$ block-diagonal matrix:

$$
W_{tile}=\mathrm{diag}(W_0,W_1,\ldots,W_7),\qquad
W_k=\begin{bmatrix}w_{k,r}&-w_{k,i}\\w_{k,i}&w_{k,r}\end{bmatrix}.
$$

The lower inputs for all 64 FFT batches are packed as a $16\times64$ matrix:

$$
X_{tile}=
\begin{bmatrix}
b_{0,r}^{(0)}&\cdots&b_{0,r}^{(63)}\\
b_{0,i}^{(0)}&\cdots&b_{0,i}^{(63)}\\
\vdots&&\vdots\\
b_{7,r}^{(0)}&\cdots&b_{7,r}^{(63)}\\
b_{7,i}^{(0)}&\cdots&b_{7,i}^{(63)}
\end{bmatrix}.
$$

One weight-stationary `tiled_matmul_auto()` call computes

$$
Y_{tile}^{int32}=W_{tile}^{Q7}X_{tile}^{Q7}.
$$

The Gemmini WS wrapper keeps a reusable operand in scratchpad when it fits and
uses internal double buffering. Thus one Gemmini operation performs the
complex twiddle rotations for eight butterflies across all 64 batches. A
partial final tile is zero padded. The matrix is intentionally sparse: it
contains 32 nonzero coefficients out of 256 entries. This preserves
$O(N\log N)$ FFT scheduling, but the unused systolic-array products are an
important reason Gemmini is not automatically efficient for this workload.

Gemmini returns full int32 accumulators. For every complex result the host then
uses the same two-step Q7 arithmetic as the CPU and RVV implementations:

$$
\begin{aligned}
\hat{t}_r &= \mathrm{sat}_{\mathrm{int8}}
  \left(\mathrm{RNU}\left(\frac{Y_r}{2^7}\right)\right), \\
\hat{t}_i &= \mathrm{sat}_{\mathrm{int8}}
  \left(\mathrm{RNU}\left(\frac{Y_i}{2^7}\right)\right), \\
u &= \mathrm{RNU}\left(\frac{a+\hat{t}}{2}\right), \\
l &= \mathrm{RNU}\left(\frac{a-\hat{t}}{2}\right).
\end{aligned}
$$

The separate rounding steps are required for bit-exact agreement. Fusing the
complete butterfly into a single Gemmini matrix followed by one shift would
change the numerical definition. Unity-twiddle butterflies ($w=1+j0$) are also
executed by the host because Q7 stores unity as 127 rather than exact 128.

The binary also validates Gemmini-native output scaling independently of the
timed FFT regions. A WS matmul with output scale $1/128$ and a WS butterfly
matrix with output scale $1/2$ are checked against the software model of
Gemmini's round-to-nearest-even scaling and saturating INT8 store. Boundary,
negative, and halfway cases are included. The native hardware/model results
must match exactly; the test separately reports how often they differ from
the FFT's current RNU definition.

For `wi=-128`, the coefficient `-wi=+128` cannot be represented by Gemmini's
signed INT8 `elem_t`. The packed matrix uses +127, then the host adds one copy
of $b_i$ to the corresponding int32 real accumulator before RNU. This exactly
restores $w_r b_r-w_i b_i$.

The execution of one FFT stage is therefore:

```text
host unity butterflies
        |
        v
enumerate non-unity butterflies in groups of 8
        |
        v
load prepacked 16x16 twiddle matrix; pack dynamic 16x64 lower-input matrix
        |
        v
Gemmini WS tiled_matmul_auto -> 16x64 int32 accumulators
        |
        v
host Q7 RNU/saturation and upper/lower add-subtract
        |
        v
next radix-2 stage
```

Thus Gemmini accelerates the batched complex twiddle multiplications, while the
host retains bit reversal, packing, exact fixed-point conversion, and final
butterfly add/subtract operations.

### Two-stage-fused Gemmini dataflow

The fused schedule processes five pairs for the 10-stage FFT:

```text
four 4-bin groups
        |
        v
pack eight first-stage lower inputs
        |
        v
Gemmini first-stage rotations
        |
        v
host RNU + saturation + scaled add/sub
        |
        v
temporary Q7 tile (no global FFT-buffer write)
        |
        v
pack second-stage lower inputs
        |
        v
Gemmini second-stage rotations
        |
        v
host RNU + saturation + scaled add/sub
        |
        v
write final outputs of the stage pair
```

Offset-zero rotations remain exact host operations in both stages. The fused
plan precomputes 512 paired-stage tiles, including separate first- and
second-stage 16x16 weight matrices and the `wi=-128` correction metadata.

## Gemmini PE configuration

The testbench does not create or change processing elements in software. PE
count is a hardware-generation parameter of the Chipyard Gemmini instance.
The software header used by this repository declares:

```c
#define DIM 16
typedef int8_t elem_t;
typedef int32_t acc_t;
```

For the standard Gemmini systolic array, `DIM=16` means 16 PE rows by 16 PE
columns, or 256 INT8 multiply-accumulate PEs. The Spike output

```text
Gemmini extension configured with:
    dim = 16
```

confirms that the simulator uses the same array dimension as the compiled
software. `gemmini_params.h` also specifies four scratchpad banks, 4,096 rows
per bank, and 1,024 accumulator rows. These software parameters must match the
Gemmini hardware/Spike configuration; changing only the C header does not
resize the actual accelerator.

The FFT tile dimensions were selected from this hardware contract:

| Quantity | Value |
| --- | ---: |
| Systolic-array dimension | 16x16 |
| Physical PE count | 256 |
| Butterflies packed per tile | 8 |
| Twiddle matrix | 16x16 INT8 |
| Batch/input matrix | 16x64 INT8 |
| Output accumulator matrix | 16x64 INT32 |

The 16x64 output requires four 16-column output tiles on a 16x16 array. Each
16x16 twiddle matrix contains only 32 nonzero coefficients, so its arithmetic
density is 12.5%. The remaining zero-valued matrix positions still occupy the
dense systolic-array schedule. Consequently, the accelerator contains 256 PEs,
but this sparse block-diagonal FFT mapping cannot keep all of them doing useful
nonzero multiplications. This is a structural limitation of mapping radix-2
butterflies onto a dense matrix engine, not an absence of configured PEs.

## Build

```sh
source ~/chipyard_1.13.0/chipyard/env.sh

./scripts/configure_build.sh \
    linux-pk gemmini fft_batched_int8_gemmini default

/usr/bin/cmake --build \
    build/cmake/linux-pk-fft_batched_int8_gemmini-gemmini-default \
    --target run-spike-gemmini
```

## Gemmini Spike

The CMake target above runs the following equivalent command:

```sh
spike --isa=rv64gc_zicntr_zihpm --extension=gemmini \
    pk build/linux-pk/fft_batched_int8_gemmini/default/gemmini/static/fft_batched_int8_gemmini
```

A valid run must finish with:

```text
mismatched complex points: 0 / 65536
FFT batched int8 Gemmini: PASS (Q7 and native-scaling validation)
```

The measured FFT region includes bit reversal, matrix packing, Gemmini matrix
multiplication, accumulator readback, and host post-processing. Twiddle-plan
cycles and input reload are reported or executed outside that region.

Sourcing Chipyard's `env.sh` selects the same RV64GC compiler, Spike, and `pk`
used by `sentence_gemmini`. The `default` profile builds RV64GC host code and
supplies Gemmini as a RoCC extension; it does not enable RVV. Tree and SLP
vectorization are disabled for the host code.

## Verified baseline and stage-fused Spike result

```text
Gemmini extension configured with:
    dim = 16
FFT size: 1024
batches: 64
Gemmini butterflies/tile: 8
twiddle plan cycles: 494232

Gemmini variant: baseline
benchmark runs: 10
average FFT cycles: 15288837
mismatched complex points: 0 / 65536
FFT int8 Gemmini baseline: PASS

Gemmini variant: stage-fused
benchmark runs: 10
average FFT cycles: 17602856
mismatched complex points: 0 / 65536
FFT int8 Gemmini stage-fused: PASS

FFT batched int8 Gemmini: PASS (Q7 and native-scaling validation)
```

Both values are complete timed FFT regions: bit reversal, matrix packing,
Gemmini execution and synchronization, int32 accumulator readback, and scalar
Q7 post-processing. They exclude the shared one-time twiddle plan and input
reload. Stage fusion is `1.15x` slower than baseline on Gemmini. Although it
removes global intermediate traffic, it changes the packing from eight
arbitrary same-stage butterflies to four coupled four-bin groups and performs
more Gemmini/host handoffs. Those costs exceed the saved store/reload traffic.
Compare these values with equivalently scoped CPU/RVV FFT regions, not with a
butterfly-only counter.

### Full Gemmini-native scaling validation

The same binary runs a third complete FFT variant using Gemmini-native
round-to-nearest-even and saturating INT8 output scaling. Twiddle matmul writes
Q7 INT8 directly with output scale $1/128$. A second WS matmul applies the
butterfly matrix

$$
\begin{bmatrix}1&1\\1&-1\end{bmatrix}
$$

with output scale $1/2$, so Gemmini performs the upper/lower add-subtract,
rounding, saturation, and stage scaling. Unity butterflies use the same
Gemmini butterfly matrix. The unrepresentable `wi=-128` coefficient remains a
host-specialized correction.

A scalar reference independently implements the same Gemmini-native RNE and
saturation semantics for all 10 stages. The verified Spike result is:

| Variant | Average FFT cycles | Reference mismatches | Result |
| --- | ---: | ---: | --- |
| native scaling | 20,090,711 | 0 / 65,536 | PASS |

This is a complete 1,024-point, 64-batch FFT comparison, not only a primitive
test. It is `1.31x` slower than the matching 15,295,181-cycle host-Q7 baseline
in this build. Moving Q7 post-processing into Gemmini is therefore
functionally valid, but two accelerator matmuls per butterfly group cost more
than the saved scalar post-processing in the current mapping.

## Verified 30 MHz Gemmini hardware result

The current baseline measurement was collected on the 30 MHz Gemmini hardware
with a 10-run average and passed bit-exact validation.

| Variant | Average FFT cycles | Time at 30 MHz | Mismatches | Result |
| --- | ---: | ---: | ---: | --- |
| baseline | **19,073,443** | **635.78 ms** | 0 / 65,536 | PASS |
| stage-fused | pending | pending | — | pending |
| native scaling | pending | pending | — | pending |

Twiddle-plan creation took 512,544 cycles, or 17.08 ms at 30 MHz, and is
excluded from the FFT average. The baseline requires about 298,023 cycles, or
9.93 ms, per FFT.

### Archived earlier 30 MHz result

The preceding hardware binary measured 19,595,333 cycles (653.18 ms) for the
baseline and 26,599,496 cycles (886.65 ms) for stage fusion. The current
baseline is 2.66% faster than that earlier baseline. The old stage-fused value
must not be combined with the current baseline to calculate a current fusion
speedup; rerun the current stage-fused binary first.

## Bare-metal RV64GC build

The bare-metal image uses `/opt/riscv_baremetal_medany`, `rv64gc/lp64d`, the
medany code model, repository startup/trap/syscall/minilib sources, and the
Gemmini RoCC API. It does not enable or contain RVV instructions.

```sh
./scripts/configure_build.sh \
    baremetal gemmini fft_batched_int8_gemmini GEMMINI

cmake --build \
    build/cmake/baremetal-fft_batched_int8_gemmini-gemmini-GEMMINI \
    --target baremetal-dump
```

Artifacts are written to:

```text
build/baremetal/fft_batched_int8_gemmini/
  GEMMINI_fft_batched_int8_gemmini_baremetal.elf
  GEMMINI_fft_batched_int8_gemmini_baremetal.bin
  GEMMINI_fft_batched_int8_gemmini_baremetal.dump
  GEMMINI_fft_batched_int8_gemmini_baremetal.map
```

## Bare-metal flash

First identify the intended unmounted whole SD-card device:

```sh
lsblk
```

Then build and flash with the top-level Make target:

```sh
make baremetal-flash \
    TESTBENCH=fft_batched_int8_gemmini \
    BACKEND=gemmini \
    HARDWARE_CONFIG=GEMMINI \
    SDCARD_DEVICE=/dev/sdX \
    SDCARD_BLOCK=34 \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole SD-card disk, not a partition. The command
rejects partitions and mounted devices. `FLASH_CONFIRM=YES` authorizes writing
the generated
`GEMMINI_fft_batched_int8_gemmini_baremetal.bin` beginning at 512-byte block
34; this overwrites data on the selected device.

To verify the selected device without writing, omit `FLASH_CONFIRM=YES`. The
flash helper will print the device information and stop before `dd`.

The repository bare-metal runtime uses board UART/MMIO and does not define the
Spike `tohost/fromhost` interface. Use the Linux/pk ELF for Spike validation
and the bare-metal binary for the Gemmini FPGA/platform boot flow.
