# 1D_CNN

## v1.1

speedup padding operation in conv1d forwarding

## v1.2

- unroll `inC` inner loop by 4 for RVV conv1d fast paths (int8 and float32)
- tail case handles remaining `inC` elements that do not fit the 4-way unroll

## v1.3

- add unroll2/unroll4/unroll8 variants for conv1d and fc RVV paths
- rename fast kernels to include the unroll factor in function names

## Operator optimizations

- conv1d (int8): RVV vectorization across outC, unroll2/4/8 variants on `inC`, tail handling for `inC % unroll`, cache blocking not applied
- conv1d (float32): RVV vectorization across outC, unroll2/4/8 variants on `inC`, tail handling for `inC % unroll`, cache blocking not applied
- fc (float32): RVV vectorization across outW, outW blocking, unroll2/4/8 variants on `inDim`, tail handling for `inDim % unroll`
- fc (int8): RVV vectorization across outW, outW blocking, unroll2/4/8 variants on `inDim`, tail handling for `inDim % unroll`
