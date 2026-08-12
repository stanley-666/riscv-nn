<!-- SPDX-FileContributor: Person: Stanley Lee -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# RVV DFT GEMV testbench

Linux/Spike and bare-metal compile the same `testbench/fft_gemv/fft_gemv.c`
source and execute the same `main()`. Bare-metal replaces only startup,
linking, minilib, and the `nn_runtime` implementation; there is no separate
application adapter.
The bare-metal startup prints hardware and RVV/VLEN information before entering
this shared `main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

This testbench expresses the 1024-point complex DFT as Conv1D-style
scalar-vector MAC operations. The original input arrays remain unchanged. Each
scalar input sample multiplies contiguous vectors of twiddle weights and
accumulates output frequency bins with `vfmacc.vf` and `vfnmsac.vf`.

All static input, ground-truth, twiddle, and output arrays are explicitly
aligned to 64-byte boundaries.

The natural twiddle layout `[output_bin][input_index]` is transposed in a
measured preprocessing step into `[input_index][output_bin]`, matching the
`(input channel, output channel)` weight layout used by the RVV Conv1D kernel.

This produces the same DFT values as the radix-2 FFT, but performs `O(N^2)`
work rather than the FFT's `O(N log N)` work.

Because each GEMV output accumulates 1024 FP32 terms in a different order from
the radix-2 FFT, results are mathematically equivalent but not bit-exact. This
test uses `atol=2e-4` and `rtol=1e-6`; batch printing retains the same
`bin N: real +/-imagj` format as the FFT testbench.

```sh
python3 py/fft/fft_groundtruth.py --size 1024 \
    --header testbench/fft_gemv/fft_vectors.h
```
```sh
./scripts/configure_build.sh linux-pk vector fft_gemv zvl128b
```
```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/fft_gemv/zvl128b/vector/static/fft_gemv
```

Build the bare-metal RVV image with:

```sh
./scripts/configure_build.sh baremetal vector fft_gemv V128D128B
```

The equivalent Makefile command is:

```sh
make baremetal \
    TESTBENCH=fft_gemv \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B
```

After identifying the intended unmounted whole SD-card device, flash with:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=fft_gemv \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole SD-card device, not a partition. Flashing
overwrites data on the selected device.

The generated ELF, binary, and map are placed under
`build/baremetal/fft_gemv/`. The 1024x1024 original and reordered complex
twiddle matrices occupy about 16 MiB of BSS in external RAM.

The output separately reports twiddle-generation, memory-reorder, GEMV-kernel,
and end-to-end cycles. Validation and bin printing are outside all measurements.

## Performance results

The result below is a single Spike run of the 1024-point FP32 complex DFT GEMV.
It passed with maximum component error `0.000145` at bin 297 and maximum
tolerance ratio `0.621375` at bin 297.

| Environment | Configuration | Twiddle generation cycles | Memory reorder cycles | GEMV cycles | Total cycles | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Spike simulation | `zvl128b`, LMUL=m8 | 252,806,463 | 13,262,858 | 438,558 | 266,507,879 | PASS |

The total includes runtime construction of the full 1024x1024 complex twiddle
matrix. Validation and bin printing are excluded. These values are from one run
rather than an average.

### Genesys2 FPGA configuration matrix

The following hardware configurations are planned for testing. A dash means
that no result has been recorded yet.

| Configuration | Twiddle generation cycles | Memory reorder cycles | GEMV cycles | Total cycles | Max component error | Max tolerance ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `GENV128D64` | — | — | — | — | — | — | Pending |
| `GENV128D128` | — | — | — | — | — | — | Pending |
| `GENV256D64` | — | — | — | — | — | — | Pending |
| `GENV256D128` | — | — | — | — | — | — | Pending |
| `GENV512D64` | — | — | — | — | — | — | Pending |
| `GENV512D128` | — | — | — | — | — | — | Pending |
| `LGVV128D128` | — | — | — | — | — | — | Pending |
| `LGVV256D128` | — | — | — | — | — | — | Pending |
| `LGVV512D128` | — | — | — | — | — | — | Pending |
