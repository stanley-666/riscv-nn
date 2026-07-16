# RVV Pooling

Explicit RVV 1D/2D pooling operators for FP32 and INT8.

Supported entry-point families include max pool, average pool, adaptive 1D max
pool, and adaptive 2D average pool. The 1D adaptive WC kernels operate directly
on width-major/channel-contiguous tensors and avoid an unnecessary transpose.

Window geometry comes from the `NNModule`. Boundary behavior and the divisor
used by average pooling must match the CPU reference. Validate all-negative max
windows, padded edges, uneven adaptive bins, channel tails, single-element
windows, and FP32/INT8 paths separately.

Cycle comparisons should exclude model construction and printing and report
VLEN, ISA, compiler, input shape, window size, and warm-up count.

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.  
SPDX-License-Identifier: Apache-2.0
