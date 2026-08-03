# ResNet-50 testbench

This testbench currently supports Linux/Spike only. Both CPU and RVV builds
compile `testbench/resnet50/resnet50.c` and execute its `main()`; no separate
application source is maintained.
They share the same exported weights and model construction. The CPU path uses
the exported NCHW input directly, while the RVV path converts that same input
to NHWC before calling its backend.

## Build and run

Generate `testbench/resnet50/resnet50_weights.h` with the export tools under
`py/resnet50/` before configuring this testbench.

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu resnet50 default
./scripts/configure_build.sh linux-pk vector resnet50 zvl128b
```

Run the CPU or RVV executable with Spike:

```sh
spike --isa=rv64gc_zicntr_zihpm \
    pk build/linux-pk/resnet50/default/cpu/static/resnet50

spike --isa=rv64gcv_zicntr_zihpm_zvbb_zvl128b_zve64d \
    pk build/linux-pk/resnet50/zvl128b/vector/static/resnet50
```

There is currently no bare-metal target for the ResNet-50 testbench, so no
bare-metal image or SD-card flash command is available.
