<!-- SPDX-FileContributor: Person: Stanley Lee -->
# CPU Normalization

This directory currently contains FP32 LayerNorm through
`layernorm1d_cpu(NNModule *, void *, void *)`.

For each width position, channels are normalized independently:

```text
mean = sum(x) / C
variance = sum((x - mean)^2) / C
y = ((x - mean) / sqrt(variance + eps)) * weight + bias
```

- Layout: `[W][C]`; normalization axis is `C`.
- Variance is population variance with denominator `C`, not `C - 1`.
- Data type: FP32 only.
- `weight` and `bias` contain `C` values; `eps` comes from the layer.
- In-place input/output is supported by the current row-wise implementation.

Compare CPU and RVV with an absolute/relative tolerance because their reduction
orders differ. Include constant rows, non-multiple channel sizes, nontrivial
weight/bias, and multiple epsilon values in validation.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
