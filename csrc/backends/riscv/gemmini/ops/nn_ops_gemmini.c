/*
 * Copyright 2026 Stanley Lee
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_gemmini_internal.h"

#include <include/gemmini.h>
#include <assert.h>

static int32_t gemmini_round_to_i32(float value)
{
    return (int32_t)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static int8_t gemmini_requantize_i8(int32_t value, float multiplier,
                                    int32_t zero_point)
{
    int32_t q = gemmini_round_to_i32((float)value * multiplier) + zero_point;
    if (q < -127)
        q = -127;
    if (q > 127)
        q = 127;
    return (int8_t)q;
}

static int8_t gemmini_activate_i8(int8_t value, ActivationType activation)
{
    if (activation == RELU && value < 0)
        return 0;
    return value;
}

static int8_t gemmini_activate_raw_i8(int8_t value,
                                      NNGemminiActivation activation)
{
    if (activation == NN_GEMMINI_ACT_RELU && value < 0)
        return 0;
    return value;
}

size_t gemmini_conv1d_im2col_bytes(const NNModule *layer)
{
    return (size_t)layer->outputShape.W * layer->inputShape.C *
           layer->params.conv.filterSize * sizeof(elem_t);
}

size_t gemmini_conv1d_packed_weight_bytes(const NNModule *layer)
{
    return (size_t)layer->outputShape.C * layer->inputShape.C *
           layer->params.conv.filterSize * sizeof(elem_t);
}

size_t gemmini_conv1d_accumulator_bytes(const NNModule *layer)
{
    return (size_t)layer->outputShape.W * layer->outputShape.C * sizeof(acc_t);
}

size_t gemmini_linear_packed_weight_bytes(const NNModule *layer)
{
    int in_dim = layer->inputShape.C * layer->inputShape.W;
    return (size_t)in_dim * layer->outputShape.W * sizeof(elem_t);
}

size_t gemmini_linear_accumulator_bytes(const NNModule *layer)
{
    return (size_t)layer->outputShape.W * sizeof(acc_t);
}

void gemmini_pack_conv1d_weights_i8(const NNModule *layer, int8_t *packed_weights)
{
    const int out_channels = layer->outputShape.C;
    const int in_channels = layer->inputShape.C;
    const int kernel = layer->params.conv.filterSize;
    const int8_t *weights = (const int8_t *)layer->params.conv.weights;

    /* NNModule stores Conv1D weights as [out_channel][in_channel][kernel]. */
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int k = 0; k < kernel; ++k) {
            for (int ic = 0; ic < in_channels; ++ic) {
                size_t src = ((size_t)oc * in_channels + ic) * kernel + k;
                size_t dst = ((size_t)k * in_channels + ic) * out_channels + oc;
                packed_weights[dst] = weights[src];
            }
        }
    }
}

void gemmini_pack_conv1d_weights_i8_raw(int out_channels, int in_channels,
                                        int kernel, const int8_t *weights,
                                        int8_t *packed_weights)
{
    /* Exported Gemmini tensors use [out_channel][kernel][in_channel]. */
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int k = 0; k < kernel; ++k) {
            for (int ic = 0; ic < in_channels; ++ic) {
                size_t src = ((size_t)oc * kernel + k) * in_channels + ic;
                size_t dst = ((size_t)k * in_channels + ic) * out_channels + oc;
                packed_weights[dst] = weights[src];
            }
        }
    }
}

void gemmini_pack_linear_weights_i8(const NNModule *layer, int8_t *packed_weights)
{
    int in_dim = layer->inputShape.C * layer->inputShape.W;
    gemmini_pack_linear_weights_i8_raw(
        layer->outputShape.W, in_dim,
        (const int8_t *)layer->params.fc.weights, packed_weights);
}

void gemmini_pack_linear_weights_i8_raw(int out_dim, int in_dim,
                                        const int8_t *weights,
                                        int8_t *packed_weights)
{

    for (int o = 0; o < out_dim; ++o)
        for (int i = 0; i < in_dim; ++i)
            packed_weights[(size_t)i * out_dim + o] = weights[(size_t)o * in_dim + i];
}

static void gemmini_im2col_1d(const NNModule *layer, const int8_t *input, int8_t *im2col)
{
    int in_width = layer->inputShape.W;
    int in_channels = layer->inputShape.C;
    int out_width = layer->outputShape.W;
    int kernel = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;
    int padding = layer->params.conv.padding;
    int cols = kernel * in_channels;

    for (int ow = 0; ow < out_width; ++ow) {
        for (int k = 0; k < kernel; ++k) {
            int iw = ow * stride + k - padding;
            for (int ic = 0; ic < in_channels; ++ic) {
                int8_t value = 0;
                if (iw >= 0 && iw < in_width)
                    value = input[(size_t)ic * in_width + iw];
                im2col[(size_t)ow * cols + k * in_channels + ic] = value;
            }
        }
    }
}

void gemmini_conv1d_i8_im2col(const NNModule *layer,
                              const int8_t *input,
                              int8_t *output,
                              NNGemminiConv1DWorkspace *workspace)
{
    assert(layer != NULL && input != NULL && output != NULL && workspace != NULL);
    assert(layer->dtype == ELEM_INT8);
    assert(workspace->im2col != NULL && workspace->packed_weights != NULL &&
           workspace->accumulators != NULL);

    int out_width = layer->outputShape.W;
    int out_channels = layer->outputShape.C;
    int cols = layer->inputShape.C * layer->params.conv.filterSize;
    const acc_t *bias = (const acc_t *)layer->params.conv.bias;
    const float *multipliers = (const float *)layer->params.conv.M;
    const int32_t *zero_points = (const int32_t *)layer->params.conv.zps;

    gemmini_im2col_1d(layer, input, workspace->im2col);
    gemmini_pack_conv1d_weights_i8(layer, workspace->packed_weights);

    tiled_matmul_auto(out_width, out_channels, cols,
        workspace->im2col, workspace->packed_weights,
        bias, workspace->accumulators,
        cols, out_channels, out_channels, out_channels,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false, 0, WS);

    for (int ow = 0; ow < out_width; ++ow) {
        for (int oc = 0; oc < out_channels; ++oc) {
            int32_t acc = workspace->accumulators[(size_t)ow * out_channels + oc];
            int8_t q = gemmini_requantize_i8(acc, multipliers[oc], zero_points[oc]);
            output[(size_t)oc * out_width + ow] =
                gemmini_activate_i8(q, layer->activation);
        }
    }
}

void gemmini_fullyconnected_i8(const NNModule *layer,
                               const int8_t *input,
                               int8_t *output,
                               const int8_t *packed_weights,
                               int32_t *accumulators)
{
    assert(layer != NULL && input != NULL && output != NULL);
    assert(packed_weights != NULL && accumulators != NULL);
    assert(layer->dtype == ELEM_INT8);

    int in_dim = layer->inputShape.C * layer->inputShape.W;
    int out_dim = layer->outputShape.W;
    const acc_t *bias = (const acc_t *)layer->params.fc.bias;
    const float *multipliers = (const float *)layer->params.fc.M;
    const int32_t *zero_points = (const int32_t *)layer->params.fc.zps;

    tiled_matmul_auto(1, out_dim, in_dim,
        input, packed_weights, bias, accumulators,
        in_dim, out_dim, out_dim, out_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false, 0, WS);

    for (int o = 0; o < out_dim; ++o) {
        int8_t q = gemmini_requantize_i8(
            accumulators[o], multipliers[o], zero_points[o]);
        output[o] = gemmini_activate_i8(q, layer->activation);
    }
}

void gemmini_conv1d_i8_im2col_nhwc(
    int batch_size, int in_width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride, int out_width,
    const int8_t *input, const int8_t *packed_weights,
    const int32_t *bias, const float *multipliers,
    const int8_t *zero_points, NNGemminiActivation activation,
    int8_t *im2col, int32_t *accumulators, int8_t *output)
{
    int cols = kernel_size * in_channels;
    for (int b = 0; b < batch_size; ++b) {
        for (int ow = 0; ow < out_width; ++ow) {
            int row = b * out_width + ow;
            for (int k = 0; k < kernel_size; ++k) {
                int iw = ow * stride + k - padding;
                for (int ic = 0; ic < in_channels; ++ic) {
                    int8_t value = 0;
                    if (iw >= 0 && iw < in_width)
                        value = input[((size_t)b * in_width + iw) * in_channels + ic];
                    im2col[(size_t)row * cols + k * in_channels + ic] = value;
                }
            }
        }
    }

    int rows = batch_size * out_width;
    tiled_matmul_auto(rows, out_channels, cols,
        im2col, packed_weights, bias, accumulators,
        cols, out_channels, out_channels, out_channels,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false, 0, WS);

    for (int row = 0; row < rows; ++row) {
        for (int oc = 0; oc < out_channels; ++oc) {
            int8_t q = gemmini_requantize_i8(
                accumulators[(size_t)row * out_channels + oc],
                multipliers[oc], zero_points[oc]);
            output[(size_t)row * out_channels + oc] =
                gemmini_activate_raw_i8(q, activation);
        }
    }
}

void gemmini_fullyconnected_i8_raw(
    int batch_size, int in_features, int out_features,
    const int8_t *input, const int8_t *packed_weights,
    const int32_t *bias, const float *multipliers,
    const int8_t *zero_points, NNGemminiActivation activation,
    int32_t *accumulators, int8_t *output)
{
    tiled_matmul_auto(batch_size, out_features, in_features,
        input, packed_weights, bias, accumulators,
        in_features, out_features, out_features, out_features,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true, false, false, true, false, 0, WS);

    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < out_features; ++o) {
            int8_t q = gemmini_requantize_i8(
                accumulators[(size_t)b * out_features + o],
                multipliers[o], zero_points[o]);
            output[(size_t)b * out_features + o] =
                gemmini_activate_raw_i8(q, activation);
        }
    }
}
