/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_pooling_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

void pool1d_cpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;
    PoolType poolType = layer->params.pool.type;
    elem_type dtype   = layer->dtype;

    (void)outC;

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *in = (float*)input;
        float *out = (float*)output;

        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                float value = (poolType == MAX_POOL) ? -INFINITY : 0.0f;
                int count = 0;
                for (int k = 0; k < poolSize; k++) {
                    int idx = pos * stride + k;
                    if (idx < inW) {
                        float v = in[c*inW + idx];
                        if (poolType == MAX_POOL) {
                            value = fmaxf(value, v);
                        } else {
                            value += v;
                            count++;
                        }
                    }
                }
                if (poolType == AVG_POOL) value = (count > 0) ? value / count : 0.0f;
                out[c*outW + pos] = value;
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t*)input;
        int8_t *out = (int8_t*)output;

        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                int8_t value = (poolType == MAX_POOL) ? INT8_MIN : 0;
                int count = 0;
                for (int k = 0; k < poolSize; k++) {
                    int idx = pos * stride + k;
                    if (idx < inW) {
                        int8_t v = in[c*inW + idx];
                        if (poolType == MAX_POOL) {
                            if (v > value) value = v;
                        } else {
                            value += v;
                            count++;
                        }
                    }
                }
                if (poolType == AVG_POOL && count > 0) value /= count;
                out[c*outW + pos] = value;
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in pool1d_forward\n");
        exit(EXIT_FAILURE);
    }
}

void pool2d_cpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NCHW layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH  = layer->inputShape.H;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;
    PoolType poolType = layer->params.pool2d.type;
    elem_type dtype   = layer->dtype;

    int paddedH = inH + 2 * padding;
    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nchw_2d(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)padded_input;
        float *out = (float *)output;
        for (int c = 0; c < inC; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                int h_start = oh * stride;
                for (int ow = 0; ow < outW; ++ow) {
                    int w_start = ow * stride;
                    float value = (poolType == MAX_POOL) ? -INFINITY : 0.0f;
                    int count = 0;
                    for (int kh = 0; kh < poolSize; ++kh) {
                        int h = h_start + kh;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            int w = w_start + kw;
                            float v = in[c * paddedH * paddedW + h * paddedW + w];
                            if (poolType == MAX_POOL) {
                                value = fmaxf(value, v);
                            } else {
                                value += v;
                                count++;
                            }
                        }
                    }
                    if (poolType == AVG_POOL) value = (count > 0) ? value / count : 0.0f;
                    out[c * outH * outW + oh * outW + ow] = value;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)padded_input;
        int8_t *out = (int8_t *)output;
        for (int c = 0; c < inC; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                int h_start = oh * stride;
                for (int ow = 0; ow < outW; ++ow) {
                    int w_start = ow * stride;
                    int8_t value = (poolType == MAX_POOL) ? INT8_MIN : 0;
                    int count = 0;
                    for (int kh = 0; kh < poolSize; ++kh) {
                        int h = h_start + kh;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            int w = w_start + kw;
                            int8_t v = in[c * paddedH * paddedW + h * paddedW + w];
                            if (poolType == MAX_POOL) {
                                if (v > value) value = v;
                            } else {
                                value += v;
                                count++;
                            }
                        }
                    }
                    if (poolType == AVG_POOL && count > 0) value /= count;
                    out[c * outH * outW + oh * outW + ow] = value;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in pool2d_cpu\n");
        exit(EXIT_FAILURE);
    }

    if (padding > 0)
        safe_free(padded_input);
}

void AdaptiveAvgPool2d_cpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NCHW layout, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH  = layer->inputShape.H;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    elem_type dtype = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *out = (float *)output;
        for (int c = 0; c < inC; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                int h_start = (oh * inH) / outH;
                int h_end = ((oh + 1) * inH) / outH;
                for (int ow = 0; ow < outW; ++ow) {
                    int w_start = (ow * inW) / outW;
                    int w_end = ((ow + 1) * inW) / outW;
                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += in[c * inH * inW + h * inW + w];
                            count++;
                        }
                    }
                    out[c * outH * outW + oh * outW + ow] = (count > 0) ? sum / count : 0.0f;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int c = 0; c < inC; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                int h_start = (oh * inH) / outH;
                int h_end = ((oh + 1) * inH) / outH;
                for (int ow = 0; ow < outW; ++ow) {
                    int w_start = (ow * inW) / outW;
                    int w_end = ((ow + 1) * inW) / outW;
                    int32_t sum = 0;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += in[c * inH * inW + h * inW + w];
                            count++;
                        }
                    }
                    out[c * outH * outW + oh * outW + ow] = (count > 0) ? (int8_t)(sum / count) : 0;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in AdaptiveAvgPool2d_cpu\n");
        exit(EXIT_FAILURE);
    }
}


void AdaptiveMaxPool1d_cpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    elem_type dtype = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *in = (float*)input;
        float *out = (float*)output;

        if (outW == 1) {
            for (int c = 0; c < inC; c++) {
                float max_val = -INFINITY;
                for (int idx = 0; idx < inW; idx++) {
                    float v = in[c*inW + idx];
                    if (v > max_val) max_val = v;
                }
                out[c] = max_val;
            }
            break;
        }

        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                int start = (pos * inW) / outW;
                int end   = ((pos + 1) * inW) / outW;
                if (end > inW) end = inW;

                float value = -INFINITY;
                for (int idx = start; idx < end; idx++) {
                    float v = in[c*inW + idx];
                    if (v > value) value = v;
                }
                out[c*outW + pos] = value;
            }
        }
        break;
    }

    case ELEM_INT8: {
        const int8_t *in = (const int8_t*)input;
        int8_t *out = (int8_t*)output;

        if (outW == 1) {
            for (int c = 0; c < inC; c++) {
                int8_t max_val = INT8_MIN;
                for (int idx = 0; idx < inW; idx++) {
                    int8_t v = in[c*inW + idx];
                    if (v > max_val) max_val = v;
                }
                out[c] = max_val;
            }
            break;
        }

        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                int start = (pos * inW) / outW;
                int end   = ((pos + 1) * inW) / outW;
                if (end > inW) end = inW;

                int8_t value = INT8_MIN;
                for (int idx = start; idx < end; idx++) {
                    int8_t v = in[c*inW + idx];
                    if (v > value) value = v;
                }
                out[c*outW + pos] = value;
            }
        }
        break;
    }

    default:
        printf("Unsupported dtype in AdaptiveMaxPool1d_forward\n");
        exit(EXIT_FAILURE);
    }
}


