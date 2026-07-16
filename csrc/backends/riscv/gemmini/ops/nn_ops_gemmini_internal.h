/*
 * Copyright 2026 Stanley Lee
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_OPS_GEMMINI_INTERNAL_H
#define NN_OPS_GEMMINI_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "nn_param.h"
#include "nn_ops_gemmini_raw.h"

typedef struct {
    int8_t *im2col;
    int8_t *packed_weights;
    int32_t *accumulators;
} NNGemminiConv1DWorkspace;

void gemmini_pack_conv1d_weights_i8(const NNModule *layer, int8_t *packed_weights);
void gemmini_pack_linear_weights_i8(const NNModule *layer, int8_t *packed_weights);

void gemmini_conv1d_i8_im2col(const NNModule *layer,
                              const int8_t *input,
                              int8_t *output,
                              NNGemminiConv1DWorkspace *workspace);

void gemmini_fullyconnected_i8(const NNModule *layer,
                               const int8_t *input,
                               int8_t *output,
                               const int8_t *packed_weights,
                               int32_t *accumulators);

size_t gemmini_conv1d_im2col_bytes(const NNModule *layer);
size_t gemmini_conv1d_packed_weight_bytes(const NNModule *layer);
size_t gemmini_conv1d_accumulator_bytes(const NNModule *layer);
size_t gemmini_linear_packed_weight_bytes(const NNModule *layer);
size_t gemmini_linear_accumulator_bytes(const NNModule *layer);

#endif
