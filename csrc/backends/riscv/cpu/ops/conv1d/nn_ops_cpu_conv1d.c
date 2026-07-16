/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_conv1d_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

static void *cpu_im2col_nchw_1d(NNModule *layer, const void *input)
{
    int outW = layer->outputShape.W;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;
    int padding = layer->params.conv.padding;
    int paddedW = inW + 2 * padding;
    int cols = inC * filterSize;
    size_t elemSize = sizeof_dtype(layer->dtype);
    const uint8_t *padded = (const uint8_t *)padded_input_create_nchw(layer, input);
    uint8_t *im2col = (uint8_t *)safe_malloc((size_t)outW * cols * elemSize);

    for (int pos = 0; pos < outW; ++pos) {
        uint8_t *row = im2col + (size_t)pos * cols * elemSize;
        int start = pos * stride;
        for (int ic = 0; ic < inC; ++ic) {
            const uint8_t *src = padded + ((size_t)ic * paddedW + start) * elemSize;
            memcpy(row + (size_t)ic * filterSize * elemSize,
                   src, (size_t)filterSize * elemSize);
        }
    }

    if (padding > 0)
        safe_free((void *)padded);
    return im2col;
}


void conv1d_cpu(NNModule *layer, void *input, void *output)
{
    // C,WS
    if (input == NULL || output == NULL || layer == NULL) {
        printf("Error: input, output, or layer is NULL.\n");
        exit(EXIT_FAILURE);
    }
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    elem_type dtype = layer->dtype;
    int cols = inC * filterSize;
    void *im2col = cpu_im2col_nchw_1d(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *input_f32 = (const float *)im2col;
        float *output_f32 = (float *)output;
        float *bias_f32 = (float *)layer->params.conv.bias;
        float *weights_f32 = (float *)layer->params.conv.weights;

        for (int pos = 0; pos < outW; ++pos) {
            const float *input_row = &input_f32[pos * cols];
            for (int oc = 0; oc < outC; ++oc) {
                const float *weight_row = &weights_f32[oc * cols];
                float sum = bias_f32[oc];
                for (int col = 0; col < cols; ++col)
                    sum += input_row[col] * weight_row[col];
                output_f32[oc * outW + pos] = activate_f32(sum, layer->activation);
            }
        }
        break;
    }

    case ELEM_INT8: {
        const int8_t *input_i8 = (const int8_t *)im2col;
        int8_t *output_i8 = (int8_t *)output;
        int32_t *bias_i32 = (int32_t *)layer->params.conv.bias;
        int8_t *weights_i8 = (int8_t *)layer->params.conv.weights;
        float *M = (float *)layer->params.conv.M;
        int32_t *Z = (int32_t *)layer->params.conv.zps;

        for (int pos = 0; pos < outW; ++pos) {
            const int8_t *input_row = &input_i8[pos * cols];
            for (int oc = 0; oc < outC; ++oc) {
                const int8_t *weight_row = &weights_i8[oc * cols];
                int32_t sum = bias_i32[oc];
                for (int col = 0; col < cols; ++col)
                    sum += (int32_t)input_row[col] * weight_row[col];
                int8_t requantized = requantize_int8_asymmetric(sum, M[oc], Z[oc]);
                output_i8[oc * outW + pos] = activate_i8(requantized, layer->activation);
            }
        }
        break;
    }

    default:
        printf("Unsupported data type in conv1d_forward\n");
        exit(EXIT_FAILURE);
    }

    safe_free(im2col);
}

