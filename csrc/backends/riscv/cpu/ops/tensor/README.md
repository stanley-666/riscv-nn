# CPU Tensor Operators

Scalar reference utilities used between neural-network layers.

- `transpose_cpu`: converts between the layer's supported CW and WC layouts.
- `save_cpu`: copies a tensor into the residual/save buffer owned by the layer.
- `add_cpu`: performs residual elementwise addition using the configured saved
  or override tensor.
- Data types: FP32 and INT8 according to the layer configuration.

Input/output buffers may not be safely aliased unless the individual operation
explicitly permits it. Validate transpose round trips, exact byte counts for
save, residual shape equality, INT8 overflow/saturation behavior, and tensor
sizes that are not multiples of common vector lengths.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
