# Kyber RVV testbench

## Build commands

From the repository root, build the Linux/Spike RVV target with:

```sh
./scripts/configure_build.sh linux-pk vector kyber_nouv_rvv zvl128b
```

Build the bare-metal RVV target with:

```sh
./scripts/configure_build.sh baremetal vector kyber V128D128B
```

The Linux and bare-metal testbench names differ because the Linux executable is
named after `kyber_nouv_rvv.c`, while the bare-metal adapter is named `kyber`.
