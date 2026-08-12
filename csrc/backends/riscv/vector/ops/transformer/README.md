<!-- SPDX-FileContributor: Person: Stanley Lee -->
# RVV Transformer Operators

This directory contains FP32 multi-head self-attention. LayerNorm is maintained
separately in `../normalization/`.

`attention1d_fp32_vpu` performs RVV Q/K/V projection, scaled QK dot products,
row Softmax, weighted value accumulation, and RVV output projection. Tensor
layout is sequence-major `[sequence][embedding]`; `embed_dim` must agree with
`num_heads * head_dim`.

After projection, keys are packed as `[head][head_dim][sequence]`. Each query
component is then broadcast as a scalar while contiguous keys and score
accumulators are processed as RVV vectors. This maps QK-transpose to GEMV and
avoids one vector reduction per query/key pair.

The score-weighted value sum keeps each context vector chunk in an RVV
accumulator across the complete sequence. The accumulator starts at zero and
is stored once, so the context buffer needs neither a preceding clear nor a
load/store pair for every source position.

The layer owns preallocated QKV, transposed-key, context, projection, and score
buffers.
Attention currently calls the scalar full-row Softmax reference for its score
vectors; it does not yet use the approximate public RVV Softmax. This difference
must be recorded when comparing attention results or cycles.

Validate each stage independently, require every Softmax row to sum to about
one, and compare final output with the CPU attention reference using FP32
tolerances. Kyber is the current end-to-end integration test.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
