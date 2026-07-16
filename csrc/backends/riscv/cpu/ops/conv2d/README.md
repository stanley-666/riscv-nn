# CPU Conv2D

Scalar reference implementation for two-dimensional convolution. The internal
entry point is `conv2d_cpu`; normal model execution selects it through the CPU
dispatcher.

- Data types: FP32 and INT8.
- Layer parameters define filter size, stride, padding, input shape, and output
  channels.
- The implementation uses an im2col-compatible computation so results can be
  compared with the RVV Conv2D kernels.
- INT8 uses int32 accumulation and per-channel requantization metadata.
- Output and input buffers must provide the number of elements described by the
  `NNModule` shapes.

Validation should cover padding corners, stride tails, channel tails, a 1x1
filter, INT8 clipping, and comparison with a framework reference. FP32 results
need tolerance-based comparison rather than `memcmp`.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
