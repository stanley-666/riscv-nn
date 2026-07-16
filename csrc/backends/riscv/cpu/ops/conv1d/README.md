# CPU Conv1D

Scalar reference implementation for one-dimensional convolution. The internal
entry point is `conv1d_cpu(NNModule *, void *, void *)`; applications normally
invoke it through the CPU inference dispatcher.

- Data types: FP32 and INT8.
- Tensor layout: width-major `[W][C]`.
- Output width follows the layer stride, padding, and filter-size parameters.
- The CPU kernel builds the im2col view and evaluates the convolution as a
  matrix-style dot product, matching the packed-kernel contract used by RVV.
- INT8 accumulates into int32, then applies per-output-channel `M` and
  zero-point parameters before saturation to int8.
- Activations are applied after accumulation and requantization as appropriate.

Use this implementation as the correctness baseline, but allow normal FP32
rounding differences when comparing it with RVV because accumulation order can
differ. Test padding boundaries, stride tails, non-multiple channel counts, and
INT8 saturation explicitly.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
