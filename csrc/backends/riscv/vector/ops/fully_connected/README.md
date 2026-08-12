<!-- SPDX-FileContributor: Person: Stanley Lee -->
# RVV Fully Connected

Explicit RVV kernels for FP32 and INT8 fully connected layers.

- Input length: `inputShape.W * inputShape.C`; output length:
  `outputShape.W`.
- Weights are transposed to `[input][output]` so each scalar input multiplies a
  contiguous output-channel vector.
- FP32 variants use LMUL m2/m4/m8 and unroll factors 2/4/8. These functions are
  named `fullyconnected_fp32_vpu_unrollX_mY`.
- INT8 widens multiplication and accumulates into int32, then performs
  per-channel requantization and saturation.
- Stable inference currently selects the m8 entry points; internal variants are
  benchmark APIs, not stable public APIs.

Simple pointwise activations may be fused into each output chunk. Stable
inference runs FC `SIGMOID` as a separate finalizer after every output has been
stored. Softmax also remains outside the fused FC kernel: the chunk callback
stores raw logits and the caller runs full-tensor Softmax. See
`../activation/README.md` for numerical behavior.

Validate matrix output separately from activation, include odd dimensions and
zero inputs, and compare all variants with the CPU implementation.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
