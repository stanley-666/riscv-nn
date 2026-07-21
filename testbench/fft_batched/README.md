# Batched RVV FFT testbench

This testbench runs 64 independent 1024-point FP32 radix-2 FFTs. Every batch
uses the same input and NumPy ground truth as `testbench/fft`.

The original batch-major input `[batch][bin]` is transposed in a measured
preprocessing step into `[bin][batch]`. RVV operations then span the batch
dimension, providing parallel work even during the early FFT stages.

The four batch-major and bin-major working arrays are explicitly aligned to
64-byte boundaries. Since one row in the bin-major layout is 64 FP32 values
(256 bytes), every row remains 64-byte aligned for RVV loads and stores.

Before entering the measured regions, the testbench rejects NaN/Inf inputs and
checks the conservative FP32 component bound
`FFT_SIZE * max(abs(real) + abs(imag)) <= FLT_MAX`. Each strip-mined loop also
rejects a zero VL or a VL larger than the remaining batch count. Validation
counts NaN/Inf outputs explicitly, so non-finite arithmetic cannot silently
bypass the normal tolerance comparisons. These checks distinguish arithmetic
overflow and invalid VL behavior from finite-but-incorrect hardware results.
Every one of the `64 * 1024` complex outputs is checked. For a point outside
the configured absolute/relative tolerance, the diagnostic includes its batch
and bin indices, actual and expected complex values, and component errors. To
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

## Performance results

The result below is a single Spike run with 64 identical 1024-point FP32 FFTs.
All batches passed with maximum component error `0.000024` at batch 0 bin 757
and maximum tolerance ratio `0.385719` at batch 0 bin 1015.

| Environment | Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, LMUL=m4 | 64 | 827,078 | 1,104,678 | 1,931,756 | 17,260 | 30,183 | PASS |
| Spike simulation | `zvl512b`, LMUL=m4 | 64 | 827,078 | 608,994 | 1,436,072 | 9,515 | 22,438 | PASS |

Input initialization, validation, and bin printing are outside the measured
regions. These values are from one run rather than an average.

### Genesys2 FPGA configuration matrix

The following nine Genesys2 FPGA configurations are included in the hardware
test matrix. A dash means that no result has been recorded yet.

| Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Max component error | Max tolerance ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | 64 | 3,263,008 | 2,828,306 | 6,091,314 | 44,192 | 95,176 | 29.644190 | 585,500.875000 | FAIL |
| `GENV128D128` | 64 | 3,214,137 | 2,330,342 | 5,544,479 | 36,411 | 86,632 | 30.909445 | 609,266.875000 | FAIL |
| `GENV256D64` | 64 | 3,261,749 | 2,550,859 | 5,812,608 | 39,857 | 90,822 | 39.438728 | 737,125.500000 | FAIL |
| `GENV256D128` | 64 | 3,214,248 | 1,926,725 | 5,140,973 | 30,105 | 80,327 | 31.713062 | 526,786.562500 | FAIL |
| `GENV512D64` | 64 | 3,264,190 | 2,314,376 | 5,578,566 | 36,162 | 87,165 | 47.745102 | 943,011.062500 | FAIL |
| `GENV512D128` | 64 | 3,214,264 | 1,661,031 | 4,875,295 | 25,953 | 76,176 | 39.449802 | 759,238.500000 | FAIL |
| `LGVV128D128` | 64 | 3,214,396 | 2,155,732 | 5,370,128 | 33,683 | 83,908 | 31.002779 | 513,741.343750 | FAIL |
| `LGVV256D128` | 64 | 3,212,166 | 1,749,850 | 4,962,016 | 27,341 | 77,531 | 29.497887 | 577,860.687500 | FAIL |
| `LGVV512D128` | 64 | 3,218,389 | 1,336,818 | 4,555,207 | 20,887 | 71,175 | 39.888527 | 566,915.125000 | FAIL |

The `LGVV128D128` failure reported its maximum component error at batch 12,
bin 459, and its maximum tolerance ratio at batch 12, bin 795. With FP32
LMUL=m4 and VLEN=128, VLMAX is 16, so batch 12 is in the upper part of one
vector.

The `LGVV256D128` failure reported its maximum component error at batch 30,
bin 459, and its maximum tolerance ratio at batch 30, bin 245. With FP32
LMUL=m4 and VLEN=256, VLMAX is 32, so batch 30 is near the top of one vector.

The `LGVV512D128` failure reported its maximum component error at batch 60,
bin 638, and its maximum tolerance ratio at batch 40, bin 126. The tolerance
was `atol=0.000050`, `rtol=0.000001`. The large error is not normal FP32
rounding error and should be investigated before treating its cycle count as a
validated performance result.

The complete hardware log identifies this run as RV64 with FLEN=64 and
VLEN=512:

```text
misa: 0x800000000034112d
XLEN: 64
FLEN: 64
RVV: enabled and detected
VLEN: 64 bytes (512 bits)
```

The displayed bins are taken from batch 0, and they agree with the NumPy
ground truth to normal FP32 precision. For example, bin 0 was
`29.847015-39.334896j`, versus ground truth
`29.847013-39.334900j`. The validation failures occur in other batches:

- Maximum component error: batch 60, bin 638.
- Maximum normalized tolerance error: batch 40, bin 126.

With FP32 LMUL=m4 and VLEN=512, one RVV operation covers all 64 batches at
once. Batch indices 40 and 60 are therefore in the upper elements of that
vector. A matching VLEN=512 Spike run passed with maximum component error
`0.000024` and maximum tolerance ratio `0.385719`, both at batch 0. This rules
out an algorithm-wide VLEN=512 indexing issue and ordinary FP32 overflow in the
software path. All three LGVV results show the same upper-element pattern:
batch 12 of 16 at VLEN128, batch 30 of 32 at VLEN256, and batches 40/60 of 64
at VLEN512. The remaining evidence points to the FPGA implementation of the
high-element or LMUL=m4 execution path rather than one specific VLEN.
Restricting the requested VL and testing m1/m2/m4 separately can narrow the
failing hardware condition.

Additional point-by-point logs identify the configurations more precisely:

- On `V128D64` (`VLEN=128`), batches 0--3 are correct while mismatches begin
  at batch 4. Incorrect results repeat in pairs, for example batches 4/5,
  6/7, and 8/9.
- On `V512D128` (`VLEN=512`), batches 0--15 are correct while mismatches begin
  at batch 16. Incorrect results repeat in groups of four, for example batches
  16--19 and 20--23.

In both cases, the number of initially correct FP32 elements is exactly
`VLEN / 32`, the capacity of one LMUL=m1 vector register. The failing kernel
requests LMUL=m4, so these observations indicate that the first physical
register in the m4 group is handled correctly and the three additional
registers are not. The repeated-result group sizes also match the respective
64-bit and 128-bit datapath widths (two and four FP32 elements). This is
evidence for an LMUL register-group/datapath lane-selection issue, not FFT
overflow or insufficient external-memory capacity.

The bit-reversal swap is currently restricted to `e32m1` strip mining as a
hardware-isolation workaround. The butterfly arithmetic remains `e32m4`. This
avoids the compiler-generated `e8,m1` plus 32-bit vector load/store sequence
(effective EMUL=m4) while preserving the m4 arithmetic path under test.

With this workaround, the `V512D128` Genesys2 run passes all 65,536 complex
points:

| Configuration | Batches | Memory reorder cycles | FFT cycles | Total cycles | FFT cycles/FFT | Total cycles/FFT | Max component error | Max tolerance ratio | Mismatches | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `V512D128` (bit reversal `e32m1`, butterfly `e32m4`) | 64 | 3,190,883 | 1,373,879 | 4,564,762 | 21,466 | 71,324 | 0.000024 | 0.385719 | 0 / 65,536 | PASS |

The passing m4 butterfly rules out a general LMUL=m4 floating-point pipeline
failure. Together with the earlier failing binary, it isolates the FPGA issue
to the compiler-generated vector-memory form with `vtype=e8,m1`, 32-bit element
loads/stores, and effective EMUL=m4. That instruction sequence is legal RVV and
passes on Spike, but it does not execute correctly on the tested FPGA design.

The recorded `LGVV128D128` console output does not contain the newer
`non-finite outputs:` diagnostic line, so it appears to have been produced by
the binary from before explicit overflow detection was added. Reflashing the
latest `fft_batched` binary is required to record that diagnostic.
