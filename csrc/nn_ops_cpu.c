#include "nn_ops_cpu.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>
#include <time.h>

void conv1d_cpu(NNModule *layer, void *input, void *output)
{
    // C,WS
    if (input == NULL || output == NULL || layer == NULL) {
        printf("Error: input, output, or layer is NULL.\n");
        exit(EXIT_FAILURE);
    }
    int outW = layer->outputShape.W;
    int outH = layer->outputShape.H;
    int outC = layer->outputShape.C;
    int inW = layer->inputShape.W;
    int inH = layer->inputShape.H;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;
    int padding = layer->params.conv.padding;
    elem_type dtype = layer->dtype;

    (void)outH;
    (void)inH;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nchw(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *input_f32 = (float *)padded_input;
        float *output_f32 = (float *)output;
        float *bias_f32 = (float *)layer->params.conv.bias;
        float *weights_f32 = (float *)layer->params.conv.weights;

        for (int oc = 0; oc < outC; oc++) {
            int base_wt_oc = oc * inC * filterSize;
            for (int pos = 0; pos < outW; pos++) {
                int start = pos * stride;
                float sum = bias_f32[oc];
                for (int ic = 0; ic < inC; ic++) {
                    int base_in = ic * paddedW;
                    int base_wt_ic = base_wt_oc + ic * filterSize;
                    for (int k = 0; k < filterSize; k++) {
                        float x = input_f32[base_in + start + k];
                        float w = weights_f32[base_wt_ic + k];
                        sum += x * w; // with fused weights will not overflow
                    }
                }
                output_f32[oc * outW + pos] = activate_f32(sum, layer->activation);
            }
        }
        break;
    }

    case ELEM_INT8: {
        int8_t *input_i8 = (int8_t *)padded_input;
        int8_t *output_i8 = (int8_t *)output;
        int32_t *bias_i32 = (int32_t *)layer->params.conv.bias;
        int8_t *weights_i8 = (int8_t *)layer->params.conv.weights;
        float *M = (float *)layer->params.conv.M;
        int32_t *Z = (int32_t *)layer->params.conv.zps;

        for (int oc = 0; oc < outC; oc++) {
            int base_wt_oc = oc * inC * filterSize;
            for (int pos = 0; pos < outW; pos++) {
                int32_t sum = bias_i32[oc];
                int start = pos * stride;
                for (int ic = 0; ic < inC; ic++) {
                    int base_in = ic * paddedW;
                    int base_wt_ic = base_wt_oc + ic * filterSize;
                    for (int k = 0; k < filterSize; k++) {
                        int8_t x = input_i8[base_in + start + k];
                        int8_t w = weights_i8[base_wt_ic + k];
                        sum += x * w;
                    }
                }
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

    if (padding > 0) {
        safe_free(padded_input);
    }
}

void conv2d_cpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NCHW layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inH = layer->inputShape.H;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    elem_type dtype = layer->dtype;

    int paddedH = inH + 2 * padding;
    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nchw_2d(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *input_f32 = (float *)padded_input;
        float *output_f32 = (float *)output;
        float *bias_f32 = (float *)layer->params.conv2d.bias;
        float *weights_f32 = (float *)layer->params.conv2d.weights;

        for (int oc = 0; oc < outC; ++oc) {
            int base_wt_oc = oc * inC * filterSize * filterSize;
            for (int oh = 0; oh < outH; ++oh) {
                int in_h = oh * stride;
                for (int ow = 0; ow < outW; ++ow) {
                    int in_w = ow * stride;
                    float sum = bias_f32[oc];
                    for (int ic = 0; ic < inC; ++ic) {
                        int base_in = ic * paddedH * paddedW;
                        int base_wt_ic = base_wt_oc + ic * filterSize * filterSize;
                        for (int kh = 0; kh < filterSize; ++kh) {
                            int in_row = base_in + (in_h + kh) * paddedW;
                            int base_wt_kh = base_wt_ic + kh * filterSize;
                            for (int kw = 0; kw < filterSize; ++kw) {
                                float x = input_f32[in_row + in_w + kw];
                                float w = weights_f32[base_wt_kh + kw];
                                sum += x * w;
                            }
                        }
                    }
                    output_f32[oc * outH * outW + oh * outW + ow] = activate_f32(sum, layer->activation);
                }
            }
        }
        break;
    }

    case ELEM_INT8: {
        int8_t *input_i8 = (int8_t *)padded_input;
        int8_t *output_i8 = (int8_t *)output;
        int32_t *bias_i32 = (int32_t *)layer->params.conv2d.bias;
        int8_t *weights_i8 = (int8_t *)layer->params.conv2d.weights;
        float *M = (float *)layer->params.conv2d.M;
        int32_t *Z = (int32_t *)layer->params.conv2d.zps;

        for (int oc = 0; oc < outC; ++oc) {
            int base_wt_oc = oc * inC * filterSize * filterSize;
            for (int oh = 0; oh < outH; ++oh) {
                int in_h = oh * stride;
                for (int ow = 0; ow < outW; ++ow) {
                    int in_w = ow * stride;
                    int32_t sum = bias_i32[oc];
                    for (int ic = 0; ic < inC; ++ic) {
                        int base_in = ic * paddedH * paddedW;
                        int base_wt_ic = base_wt_oc + ic * filterSize * filterSize;
                        for (int kh = 0; kh < filterSize; ++kh) {
                            int in_row = base_in + (in_h + kh) * paddedW;
                            int base_wt_kh = base_wt_ic + kh * filterSize;
                            for (int kw = 0; kw < filterSize; ++kw) {
                                int8_t x = input_i8[in_row + in_w + kw];
                                int8_t w = weights_i8[base_wt_kh + kw];
                                sum += x * w;
                            }
                        }
                    }
                    int8_t rq = requantize_int8_asymmetric(sum, M[oc], Z[oc]);
                    output_i8[oc * outH * outW + oh * outW + ow] = activate_i8(rq, layer->activation);
                }
            }
        }
        break;
    }

    default:
        printf("Unsupported dtype in conv2d_cpu\n");
        exit(EXIT_FAILURE);
    }

    if (padding > 0)
        safe_free(padded_input);
}

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

void fullyconnected_cpu(NNModule *layer, void *input, void *output)
{
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    
    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *input_f32 = (float*)input;
        float *output_f32 = (float*)output;
        const float *weights_f32 = (const float*)layer->params.fc.weights;
        const float *bias_f32 = (const float*)layer->params.fc.bias;
        ActivationType act = layer->activation;

        if (act == SOFTMAX) {
            printf("Softmax activation in FC layer\n");
            for (int o = 0; o < outW; o++) {
                float sum = bias_f32[o];
                for (int i = 0; i < inDim; i++)
                    sum += input_f32[i] * weights_f32[o * inDim + i];
                output_f32[o] = sum; // logits
            }
            softmax_f32(output_f32, output_f32, outW);
        } else {
            for (int o = 0; o < outW; o++) {
                float sum = bias_f32[o];
                for (int i = 0; i < inDim; i++)
                    sum += input_f32[i] * weights_f32[o * inDim + i];
                output_f32[o] = activate_f32(sum, act);
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *input_i8 = (int8_t*)input;
        int8_t *output_i8 = (int8_t*)output;
        const int8_t *weights_i8 = (const int8_t*)layer->params.fc.weights;
        const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
        const float *M = (const float*)layer->params.fc.M;
        const int32_t *Z = (const int32_t*)layer->params.fc.zps;

        for (int o = 0; o < outW; o++) {
            int32_t sum = bias_i32[o];
            for (int i = 0; i < inDim; i++) {
                sum += input_i8[i] * weights_i8[o * inDim + i];
            }
            int8_t requantized = requantize_int8_asymmetric(sum, M[o], Z[o]);
            output_i8[o] = activate_i8(requantized, layer->activation);
        }
        break;
    }
    default:
        printf("Unsupported dtype in fc_forward\n");
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

void transpose_cpu(NNModule *layer, void *input, void *output)
{
    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    TransposeType mode = layer->params.transpose.mode;

    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *out = (float *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c)
                for (int w = 0; w < width; ++w)
                    out[w * channels + c] = in[c * width + w];
        } else {
            for (int w = 0; w < width; ++w)
                for (int c = 0; c < channels; ++c)
                    out[c * width + w] = in[w * channels + c];
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c)
                for (int w = 0; w < width; ++w)
                    out[w * channels + c] = in[c * width + w];
        } else {
            for (int w = 0; w < width; ++w)
                for (int c = 0; c < channels; ++c)
                    out[c * width + w] = in[w * channels + c];
        }
        break;
    }
    default:
        printf("Unsupported dtype in transpose_forward\n");
        exit(EXIT_FAILURE);
    }
}

void save_cpu(NNModule *layer, void *input, void *output)
{
    if (!layer->params.save.buffer || layer->params.save.bytes == 0) {
        printf("Save layer buffer not initialized (set buffer externally)\n");
        exit(EXIT_FAILURE);
    }
    memcpy(layer->params.save.buffer, input, layer->params.save.bytes);
    memcpy(output, input, layer->params.save.bytes);
}

void add_cpu(NNModule *layer, void *input, void *output)
{
    if (!layer->params.add.skip || layer->params.add.bytes == 0) {
        printf("Add layer skip buffer not initialized\n");
        exit(EXIT_FAILURE);
    }
    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *skip = (float *)layer->params.add.skip;
        float *out = (float *)output;
        int len = (int)(layer->params.add.bytes / sizeof(float));
        for (int i = 0; i < len; ++i)
            out[i] = activate_f32(in[i] + skip[i], layer->activation);
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *skip = (int8_t *)layer->params.add.skip;
        int8_t *out = (int8_t *)output;
        int len = (int)(layer->params.add.bytes / sizeof(int8_t));
        for (int i = 0; i < len; ++i)
            out[i] = activate_i8((int8_t)(in[i] + skip[i]), layer->activation);
        break;
    }
    default:
        printf("Unsupported dtype in add_cpu\n");
        exit(EXIT_FAILURE);
    }
}
