<!-- SPDX-FileContributor: Person: Stanley Lee -->
# RVV Conv1D

Explicit RVV FP32 and INT8 one-dimensional convolution kernels. Public model
execution uses the stable entry points selected by `nn_ops.h`; the many symbols
in the internal headers are architecture experiments and may change.

- Tensor layout: width-major `[W][C]`.
- Optimized kernels use im2col-compatible input ordering and transposed/packed
  weights so a scalar input can multiply a contiguous output-channel vector.
- FP32 uses RVV FMA accumulation. INT8 widens int8 input/weights through int16
  vectors and accumulates into int32 vectors before requantization.
- Variants cover LMUL m2/m4/m8, unroll 2/4/8, multiple accumulators, weight
  reuse, masking, and branch-removal experiments.
- The default FP32 inference path is the im2col unroll-8, two-accumulator m8
  implementation. The stable INT8 path uses the selected im2col m8 API.
- Packed weights and im2col workspaces are prepared by model/runtime code; they
  must not be replaced with the original weight layout.

Validate every experimental variant against the CPU Conv1D reference using the
same packed parameters. Record VLEN, LMUL, compiler, ISA string, layout, and
whether caches were warmed when publishing cycles.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
