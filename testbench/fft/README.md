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
make baremetal TESTBENCH=fft BACKEND=vector HARDWARE_CONFIG=V128D128B
```

After verifying the target is the intended unmounted whole SD-card device,
flash with:

```sh
make baremetal-flash TESTBENCH=fft BACKEND=vector \
    HARDWARE_CONFIG=V128D128B SDCARD_DEVICE=/dev/sdc FLASH_CONFIRM=YES
```
