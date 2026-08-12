<!-- SPDX-FileContributor: Person: Stanley Lee -->
# CPU Fully Connected

Scalar reference implementation for `y = Wx + b`. The internal entry point is
`fullyconnected_cpu`; applications normally use the model inference API.

- Data types: FP32 and INT8.
- Input length is `inputShape.W * inputShape.C`.
- Output length is `outputShape.W`.
- FP32 applies the selected pointwise activation after each complete dot
  product.
- INT8 accumulates into int32 and requantizes with per-output `M` and
  zero-point arrays.
- Full-tensor Softmax must only run after every output logit is available; it
  must not be normalized independently per output block.

Validate bias-only input, zero input, odd input/output sizes, saturation, and
the selected activation independently from matrix multiplication.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
