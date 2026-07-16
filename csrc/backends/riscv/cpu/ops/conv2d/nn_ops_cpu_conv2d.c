/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_conv2d_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

static void *cpu_im2col_nchw_2d(NNModule *layer, const void *input)
{
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH = layer->inputShape.H;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    int paddedH = inH + 2 * padding;
    int paddedW = inW + 2 * padding;
    int cols = inC * filterSize * filterSize;
    size_t elemSize = sizeof_dtype(layer->dtype);
    const uint8_t *padded = (const uint8_t *)padded_input_create_nchw_2d(layer, input);
    uint8_t *im2col = (uint8_t *)safe_malloc((size_t)outH * outW * cols * elemSize);

    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            uint8_t *row = im2col + (size_t)(oh * outW + ow) * cols * elemSize;
            for (int ic = 0; ic < inC; ++ic) {
                for (int kh = 0; kh < filterSize; ++kh) {
                    size_t srcIndex = ((size_t)ic * paddedH + oh * stride + kh) * paddedW
                                    + ow * stride;
                    size_t colIndex = ((size_t)ic * filterSize + kh) * filterSize;
                    memcpy(row + colIndex * elemSize,
                           padded + srcIndex * elemSize,
                           (size_t)filterSize * elemSize);
                }
            }
        }
    }

    if (padding > 0)
        safe_free((void *)padded);
    return im2col;
}


void conv2d_cpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NCHW layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    elem_type dtype = layer->dtype;
    int cols = inC * filterSize * filterSize;
    int rows = outH * outW;
    void *im2col = cpu_im2col_nchw_2d(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *input_f32 = (const float *)im2col;
        float *output_f32 = (float *)output;
        float *bias_f32 = (float *)layer->params.conv2d.bias;
        float *weights_f32 = (float *)layer->params.conv2d.weights;

        for (int row = 0; row < rows; ++row) {
            const float *input_row = &input_f32[row * cols];
            for (int oc = 0; oc < outC; ++oc) {
                const float *weight_row = &weights_f32[oc * cols];
                float sum = bias_f32[oc];
                for (int col = 0; col < cols; ++col)
                    sum += input_row[col] * weight_row[col];
                output_f32[oc * rows + row] = activate_f32(sum, layer->activation);
            }
        }
        break;
    }

    case ELEM_INT8: {
        const int8_t *input_i8 = (const int8_t *)im2col;
        int8_t *output_i8 = (int8_t *)output;
        int32_t *bias_i32 = (int32_t *)layer->params.conv2d.bias;
        int8_t *weights_i8 = (int8_t *)layer->params.conv2d.weights;
        float *M = (float *)layer->params.conv2d.M;
        int32_t *Z = (int32_t *)layer->params.conv2d.zps;

        for (int row = 0; row < rows; ++row) {
            const int8_t *input_row = &input_i8[row * cols];
            for (int oc = 0; oc < outC; ++oc) {
                const int8_t *weight_row = &weights_i8[oc * cols];
                int32_t sum = bias_i32[oc];
                for (int col = 0; col < cols; ++col)
                    sum += (int32_t)input_row[col] * weight_row[col];
                int8_t rq = requantize_int8_asymmetric(sum, M[oc], Z[oc]);
                output_i8[oc * rows + row] = activate_i8(rq, layer->activation);
            }
        }
        break;
    }

    default:
        printf("Unsupported dtype in conv2d_cpu\n");
        exit(EXIT_FAILURE);
    }

    safe_free(im2col);
}


