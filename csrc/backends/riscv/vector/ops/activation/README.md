# RVV Activation Operators

This directory contains the RISC-V Vector (RVV) activation kernels. Some
operators use approximations instead of the scalar C math-library functions.
This document defines their numerical behavior so that API users can reproduce
validation results and distinguish expected approximation error from an
implementation defect.

## Operator status

| Activation | FP32 RVV implementation | INT8 implementation | Notes |
| --- | --- | --- | --- |
| `NONE` | Vector store | Vector requantize and store | Exact apart from normal FP32/quantization behavior. |
| `RELU` | RVV maximum and store | RVV requantize, clamp, and store | Pointwise. |
| `LEAKY_RELU` | RVV multiply, merge, and store | Scalar activation after requantization | The negative slope is `0.01`. |
| `SIGMOID` | Degree-9 RVV polynomial | Scalar `expf` path after requantization | The validated Q4.3/Q0.7 fixed-point design is not integrated yet. |
| `TANH` | Scalar `tanhf` fallback per RVV chunk | Scalar activation after requantization | Not a fully vectorized implementation. |
| `GELU` | Scalar `erff` fallback per RVV chunk | Scalar activation after requantization | Not a fully vectorized implementation. |
| `SOFTMAX` | Full-tensor RVV operator | Unsupported | A chunk callback only stores logits; it does not normalize them. |

## FP32 Sigmoid

The mathematical reference is

```text
sigmoid(x) = 1 / (1 + exp(-x)).
```

The FP32 RVV kernels do not call `expf`. They evaluate a degree-9
least-squares polynomial with Horner FMA. The polynomial is fitted to
`sigmoid(a)` for `a` in `[0, 8]`. Negative inputs use

```text
sigmoid(-x) = 1 - sigmoid(x).
```

The absolute input is clamped to 8 before polynomial evaluation. Therefore,
values outside `[-8, 8]` intentionally saturate to the approximation at the
interval boundary instead of approaching exact mathematical zero or one.

Implementation properties:

- LMUL variants: m2, m4, and m8.
- Nine RVV FMA evaluation steps.
- No scalar loop, `expf`, division, or temporary memory round trip.
- `P(0)` is constrained to exactly `0.5`.
- Offline validation used 65,537 uniformly spaced samples in `[-16, 16]`.
- Maximum absolute error in that validation is less than `5.1e-4`.

Example from the sentence FP32 model:

```text
raw logit                 = 11.35523796
RVV approximation         = 0.99954057
expf reference            = 0.99998832
absolute error            = 0.00044775
```

This difference is expected because the RVV input is clamped to 8. Consumers
that require correctly rounded or libm-equivalent results must use the scalar
reference rather than this approximation.

## FP32 Softmax

The public full-tensor API is

```c
void softmax_f32_rvv(const float *input, float *output, size_t length);
```

`input` and `output` may point to the same buffer. Softmax is

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x))).
```

It cannot be evaluated independently by an FC output chunk because both the
maximum and denominator are global across all output channels. The per-chunk
`SOFTMAX` activation callbacks consequently store raw logits only. A caller or
operator finalizer must call `softmax_f32_rvv` after every output logit has been
stored.

The current implementation makes three strip-mined passes:

1. Load all logits and use RVV reduction to find the global maximum.
2. Reload logits, approximate `exp(x - max)`, store the exponentials, and
   reduce their global sum.
3. Reload the stored exponentials, multiply by the reciprocal sum, and store
   final probabilities.

The exponential approximation is a degree-9 polynomial fitted on `[-8, 0]`.
After maximum subtraction, values less than `-8` are treated as zero. The
offline maximum absolute exponential error on the fitted interval is
`5.1e-5`.

Spike validation of the four-class gesture model produced:

| Input | Probability sum | Maximum absolute error versus `expf` |
| --- | ---: | ---: |
| gesture 1 | 1.00000000 | 0.00001082 |
| gesture 2 | 1.00000000 | 0.00008846 |
| gesture 3 | 1.00000000 | 0.00001237 |

The classification result matched the scalar reference for all three inputs.
On that Spike configuration, a four-element Softmax averaged 131 cycles for
RVV and 280 cycles for the scalar `expf` reference over 100 measured runs after
five warm-up runs. These cycle counts are configuration-specific and must not
be treated as hardware performance guarantees.

At present, gesture testbenches call the full-tensor Softmax explicitly after
model inference. The FC chunk callback alone does not complete Softmax. Until
the FC activation finalizer is integrated, any other caller that selects
`SOFTMAX` must also make this explicit full-tensor call.

## Proposed INT8 fixed-point Sigmoid

The existing INT8 Sigmoid first requantizes the accumulator and then calls the
scalar activation implementation. It currently produces an integer result,
normally binary 0 or 1, and does not preserve a fractional probability.

A fixed-point RVV design has been validated offline but is not integrated into
the operator yet. Its required numerical contract is:

```text
input logit format : signed Q4.3
input scale        : 0.125
polynomial domain  : [-8, 8], clamped at the boundaries
output format      : Q0.7 stored in int8_t range [0, 127]
probability        : output / 128
classification     : output >= 64
```

The degree-9 polynomial uses the normalized variable

```text
t = abs(x) / 8, where t is in [0, 1].
```

It is evaluated in Q12 with these integer coefficients, in Horner order:

```c
static const int32_t sigmoid_q12_coefficients[10] = {
     48740, -198425, 293982, -139975, -103973,
    164172,  -72956,   2356,    8126,    2048
};
```

Each step uses a signed rounded Q12 multiplication:

```text
y = round((y * t) / 4096) + coefficient.
```

Exhaustive validation covered all 129 Q4.3 inputs from `-8` through `8` at
increments of `0.125`:

- Maximum probability absolute error: `0.00048871` at `x = +/-7.625`.
- Maximum Q0.7 difference: one least-significant bit.
- Q0.7 results matched the reference at 128 of 129 input points.
- The only Q0.7 mismatch was at `x = 1.25`: fixed-point 100 versus reference
  99, with float absolute error `0.00004389`.
- Maximum intermediate product: 613,109,760, safely below `INT32_MAX`.
- All binary decisions at the 0.5 threshold matched the exact reference.

Representative errors are:

| x | Fixed-point | `expf` reference | Signed error |
| ---: | ---: | ---: | ---: |
| -8.0 | 0.00024414 | 0.00033535 | -0.00009121 |
| -4.0 | 0.01806641 | 0.01798621 | +0.00008020 |
| -2.0 | 0.11914062 | 0.11920292 | -0.00006230 |
| -1.0 | 0.26879883 | 0.26894142 | -0.00014259 |
| 0.0 | 0.50000000 | 0.50000000 | 0.00000000 |
| 1.0 | 0.73120117 | 0.73105858 | +0.00014259 |
| 2.0 | 0.88085938 | 0.88079708 | +0.00006230 |
| 4.0 | 0.98193359 | 0.98201379 | -0.00008020 |
| 8.0 | 0.99975586 | 0.99966465 | +0.00009121 |

The current sentence INT8 model has `fc2_M = 0.00297849` and zero-point 0.
Those values do not establish a Q4.3 logit scale. The fixed-point kernel must
not be enabled for that model until the exporter or model builder explicitly
guarantees `logit_scale = 0.125`. The current 100-sample Spike baseline using
the existing scalar/binary behavior achieved 97/100 correct predictions; that
classification result is not proof that its intermediate values are Q4.3.

## Validation guidance

When comparing an RVV activation with a framework or libm reference:

1. Compare the same pre-activation input values.
2. Apply the documented clamp before deciding whether an error is unexpected.
3. Report maximum absolute error and, for quantized output, error in LSBs.
4. For Softmax, also verify that the complete tensor sums to approximately 1.
5. Do not validate Softmax independently per RVV chunk.
6. Keep reference evaluation and printing outside the benchmark cycle region.
7. For INT8, record input scale, input zero-point, output scale, output
   zero-point, rounding mode, and saturation range.

## License

Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.

SPDX-License-Identifier: Apache-2.0
