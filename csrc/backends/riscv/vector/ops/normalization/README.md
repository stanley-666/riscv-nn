<!-- SPDX-FileContributor: Person: Stanley Lee -->
# RVV Normalization

This directory contains `layernorm1d_fp32_vpu`. LayerNorm is separate from the
Transformer directory because normalization is useful outside attention models.

For every row in `[W][C]`, the kernel computes population mean and variance
across channels, then applies FP32 affine weight and bias:

```text
y = ((x - mean) / sqrt(variance + eps)) * weight + bias
```

The implementation uses three strip-mined m8 passes: mean reduction, variance
reduction, and normalize/affine/store. It supports channel counts larger than
VL and tails that are not multiples of VL. In-place operation is supported.

RVV reductions have a different addition order from the scalar CPU reference,
so bit-exact equality is not required. Validate maximum absolute/relative
error, output mean/variance before affine transformation, constant rows,
channel tails, and nontrivial weight/bias. Kyber provides an end-to-end check,
but a standalone LayerNorm test remains desirable.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
