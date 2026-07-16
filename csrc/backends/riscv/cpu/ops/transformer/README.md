# CPU Transformer Operators

This directory contains the scalar FP32 multi-head self-attention reference.
LayerNorm is intentionally located in `../normalization/`.

`attention1d_cpu` performs Q/K/V projection, scaled dot-product attention,
full-vector Softmax for every query/head, weighted value accumulation, and the
output projection. Layout is sequence-major `[sequence][embedding]`.

The layer must provide compatible `embed_dim`, `num_heads`, and `head_dim`, as
well as preallocated QKV, context, projection, and score buffers. The current
Softmax reference uses `expf` and subtracts the row maximum for stability.

Validation should compare projections, attention scores, row Softmax sums,
context vectors, and final output separately. Test multiple heads, a sequence
length of one, and nontrivial masks if masking support is added later.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
