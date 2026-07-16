# RVV Conv2D

Explicit RVV Conv2D kernels for FP32 and INT8.

- Stable APIs are declared in `header/nn_ops.h`; experimental declarations are
  internal to this directory.
- Optimized entry points use im2col so convolution windows become matrix-style
  dot products over contiguous output-channel vectors.
- INT8 accumulates into int32 and uses per-channel scale/zero-point metadata.
- FP32 and INT8 buffers, shapes, padding, stride, and packed weights must match
  the `NNModule` contract.

Validate padding corners, output shape calculations, 1x1 filters, channel and
width tails, INT8 saturation, and activation post-processing against the CPU
reference. Performance reports must state VLEN, ISA, compiler, and warm-up
policy.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
