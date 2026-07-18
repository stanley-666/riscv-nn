# ResNet-50 testbench

## Build commands

Generate `testbench/resnet50/resnet50_weights.h` with the export tools under
`py/resnet50/` before configuring this testbench.

From the repository root, build the Linux/Spike CPU and RVV targets with:

```sh
./scripts/configure_build.sh linux-pk cpu resnet50 default
./scripts/configure_build.sh linux-pk vector resnet50 zvl128b
```

There is currently no bare-metal adapter for the ResNet-50 testbench.
