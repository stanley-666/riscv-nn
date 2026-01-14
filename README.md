# RISC-V 1D CNN Inference

This repository provides a C-based 1D CNN inference stack targeting RISC-V,
with both scalar CPU and vector (RVV/VPU) execution paths, plus testbenches
and Python utilities for model preparation.

## Architecture Overview

- Model API and layer definitions live in `header/` and expose a PyTorch-like
  naming style for building 1D CNN graphs in C.
- Core inference code is in `csrc/`, split into layer logic, utilities, and
  backend-specific kernels (CPU and VPU/RVV).
- Testbenches in `testbench/` provide end-to-end inference runs for different
  models and data types, and include Spike performance results per test.
- Python utilities in `py/` support training, calibration, or data prep for
  specific models (e.g., sentence or ResNet variants).
- `gemmini/` contains Gemmini integration and related software for accelerator
  experiments.

## Repository Layout

```
.
├── csrc/               # C source: inference logic and kernels (CPU/RVV)
├── header/             # Public headers and model/layer API
├── testbench/          # End-to-end model testbenches + Spike results
├── py/                 # Training/calibration/data utilities
└── gemmini/            # Gemmini integration and tooling
```

## Testbench Notes

Each testbench directory includes its own `README.md` with Spike performance
tables. Use those files to record metrics such as cycles, CPI, and runtime.
