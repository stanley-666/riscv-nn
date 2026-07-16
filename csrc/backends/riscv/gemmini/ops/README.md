# Gemmini operators

Copyright 2026 Stanley Lee
SPDX-License-Identifier: Apache-2.0

This directory contains the RISC-V Gemmini operator adapter shared by Linux
with Spike `pk` and the repository bare-metal runtime. Platform startup,
allocation, console, and linker code do not belong in this backend.

## Dependency boundary

The low-level Gemmini command, tiling, parameter, counter, and RoCC encoding
headers live once under `csrc/backends/riscv/gemmini/ops/include/`, with the
RoCC instruction encoding under `ops/rocc-software/src/`. Those files originate
from the Gemmini/RoCC software ecosystem and retain their own
copyright and license notices. The adapter in this directory is project-owned
Apache-2.0 code.

The `ops/bareMetalC/` directory contains generated model headers, including
the latest per-tensor metadata and native HWIO convolution weights.
The executable inference entry lives in `testbench/sentence_gemmini/`, so the
backend is not responsible for a particular program entry point.
The low-level headers are maintained once in this backend and must not be
duplicated elsewhere.

## Current common API

`gemmini_conv1d_i8_im2col()` implements the repository Conv1D tensor contract:

- input layout: channel-major `[input_channel][input_width]`;
- source weight layout: `[output_channel][input_channel][kernel]`;
- packed Gemmini B matrix: `[kernel * input_channel][output_channel]`;
- output layout: channel-major `[output_channel][output_width]`;
- accumulation: INT32 Gemmini accumulator output;
- requantization: existing per-output-channel `M` and zero-point arrays.

`gemmini_fullyconnected_i8()` uses the corresponding packed
`[input][output]` B matrix. Both functions accept caller-owned aligned
workspace. They never call `malloc`, so Linux and bare-metal can use their own
runtime allocators or static buffers.

The raw API in `nn_ops_gemmini_raw.h` supports generated Gemmini model data
without depending on `NNModule`:

- `gemmini_pack_conv1d_weights_i8_raw()` reads
  `[output_channel][kernel][input_channel]` weights;
- `gemmini_conv1d_i8_im2col_nhwc()` reads and writes
  `[batch][width][channel]` tensors;
- `gemmini_pack_linear_weights_i8_raw()` converts `[output][input]` into the
  `[input][output]` Gemmini B matrix;
- all im2col, accumulator, packed-weight, and output buffers are caller-owned.

Do not interchange the raw generated-data layout with the NNModule layout.
Their Conv1D source-weight dimensions are ordered differently even though both
paths produce the same packed Gemmini matrix.

The im2col functions remain the per-channel correctness/fallback API. The
default SentenceCNN runner now uses the latest native convolution path with
HWIO weights and a single hardware store scale. These are deliberately
distinct variants because their weight and quantization contracts differ.

## Validation

Validation references:

- native/per-tensor default on Spike/pk: INT8 logit `23`, correct
  classification;
- native/per-tensor default on Gemmini hardware: INT8 logit `23`, correct
  classification, `1,682,812` end-to-end cycles at 50 MHz.

The Spike and hardware cycle counts are not directly comparable as performance
measurements. Spike is used for functional ISA validation; the FPGA cycle
counter is the hardware performance reference.
