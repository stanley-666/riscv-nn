<!-- SPDX-FileContributor: Person: Stanley Lee -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Kyber RVV testbench

Linux/Spike and bare-metal compile the same
`testbench/kyber/kyber_nouv_rvv.c` source and execute the same `main()`.
Bare-metal replaces only startup, linking, minilib, and the `nn_runtime`
implementation; there is no separate application adapter.
The bare-metal startup prints hardware and RVV/VLEN information before entering
this shared `main()`.
Extension results are diagnostic only; missing or unknown entries are reported
and execution continues.

## Build and run

From the repository root, build the Linux/Spike RVV target with:

```sh
./scripts/configure_build.sh linux-pk vector kyber_nouv_rvv zvl128b
```

Run it with Spike:

```sh
spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/kyber_nouv_rvv/zvl128b/vector/static/kyber_nouv_rvv
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector kyber V128D128B
```

Flash it to an SD card:

```sh
lsblk

make baremetal-flash \
    TESTBENCH=kyber \
    BACKEND=vector \
    HARDWARE_CONFIG=V128D128B \
    SDCARD_DEVICE=/dev/sdX \
    FLASH_CONFIRM=YES
```

Replace `/dev/sdX` with the whole, unmounted SD-card device. The flash target
rejects partitions and mounted devices, then writes the binary beginning at
512-byte block 34. This operation overwrites data on the selected device.

The Linux and bare-metal build names differ because the Linux executable is
named `kyber_nouv_rvv`, while the bare-metal build target is named `kyber`.
