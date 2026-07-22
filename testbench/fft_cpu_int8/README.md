# Pure CPU scaled int8 FFT

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

spike --isa=rv64gc_zicntr_zihpm pk \
    build/linux-pk/fft_cpu_int8/default/cpu/static/fft_cpu_int8
```

Compiler auto-vectorization is disabled, so this remains a scalar baseline.

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

The testbench reports plan, input layout, FFT, reused-plan total, and first-run
total cycles using the same boundaries as the RVV testbench. Validation checks
all 65,536 complex points for exact equality and prints the first 64 mismatches.

## Performance results

The following rows are single cold-start runs of the current pure CPU
implementation rather than multi-run averages:

| Environment | FFT size | Batches | Scale | Twiddle plan cycles | Input layout cycles | FFT cycles | Reused-plan total cycles | First-run total cycles | FFT cycles/FFT | Reused-plan cycles/FFT | First-run cycles/FFT | Mismatches | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Spike simulation, RV64GC pure CPU (RVV disabled) | 1,024 | 64 | $1/1024$ | 111,272 | 651,367 | 20,973,008 | 21,624,375 | 21,735,647 | 327,703 | 337,880 | 339,619 | 0 / 65,536 | PASS |
| Genesys2 FPGA, `GENV128D64`, pure CPU (RVV disabled), 50 MHz, cold cache | 1,024 | 64 | $1/1024$ | 193,105 | 555,448 | 13,991,874 | 14,547,322 | 14,740,427 | 218,623 | 227,301 | 230,319 | 0 / 65,536 | PASS |

The maximum component error is zero under the bit-exact Q7 definition.
