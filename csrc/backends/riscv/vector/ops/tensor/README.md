# RVV Tensor Operators

Explicit RVV data-movement and residual utilities.

- `transpose_vpu`: converts between supported CW and WC tensor layouts.
- `save_vpu`: stores a tensor for a later residual connection.
- `add_vpu`: performs elementwise residual addition.
- Data types: FP32 and INT8 according to the layer configuration.

These operations are often memory-bandwidth bound. Correctness validation must
check byte counts, layout round trips, residual shape compatibility, aliasing,
INT8 overflow/saturation semantics, and strip-mining tails. Performance reports
should identify whether source and destination are cache-warm and aligned.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
