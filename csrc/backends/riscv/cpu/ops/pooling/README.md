<!-- SPDX-FileContributor: Person: Stanley Lee -->
# CPU Pooling

Scalar reference implementations for 1D/2D max and average pooling, adaptive
1D max pooling, and adaptive 2D average pooling.

- Data types: FP32 and INT8 where exposed by the layer dispatcher.
- Entry points: `pool1d_cpu`, `pool2d_cpu`, `AdaptiveMaxPool1d_cpu`, and
  `AdaptiveAvgPool2d_cpu`.
- Window geometry is defined by pool size, stride, padding, and output shape.
- Max pooling must initialize from the first valid element or the correct
  minimum value; padded elements must not incorrectly win.
- Average-pooling validation must confirm whether the divisor counts only valid
  elements at padded boundaries.

Test boundary-only windows, windows with all negative values, non-divisible
dimensions, single-element windows, and adaptive bins with uneven sizes.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
