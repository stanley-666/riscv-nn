/*
 * Copyright 2026 Stanley Lee
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_OPS_GEMMINI_RAW_H
#define NN_OPS_GEMMINI_RAW_H

#include <stdint.h>

typedef enum {
    NN_GEMMINI_ACT_NONE = 0,
    NN_GEMMINI_ACT_RELU = 1
} NNGemminiActivation;

/* Source weights: [out_channel][kernel][in_channel].
 * Packed weights: [kernel * in_channel][out_channel]. */
void gemmini_pack_conv1d_weights_i8_raw(int out_channels, int in_channels,
                                        int kernel_size, const int8_t *weights,
                                        int8_t *packed_weights);
void gemmini_pack_linear_weights_i8_raw(int out_features, int in_features,
                                        const int8_t *weights,
                                        int8_t *packed_weights);

/* Input/output tensors use NHWC-style [batch][width][channel] storage. */
void gemmini_conv1d_i8_im2col_nhwc(
    int batch_size, int in_width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride, int out_width,
    const int8_t *input, const int8_t *packed_weights,
    const int32_t *bias, const float *multipliers,
    const int8_t *zero_points, NNGemminiActivation activation,
    int8_t *im2col, int32_t *accumulators, int8_t *output);

void gemmini_fullyconnected_i8_raw(
    int batch_size, int in_features, int out_features,
    const int8_t *input, const int8_t *packed_weights,
    const int32_t *bias, const float *multipliers,
    const int8_t *zero_points, NNGemminiActivation activation,
    int32_t *accumulators, int8_t *output);

#endif
