/* SPDX-FileContributor: Person: Stanley Lee */
#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_pool2d_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void maxpool2d_int8_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);

    const int8_t *in = (const int8_t *)padded_input;
    int8_t *out = (int8_t *)output;
    for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
        for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e8m8(inC - c);
                vint8m8_t vmaxv = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);
                const int8_t *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                for (int kh = 0; kh < poolSize; ++kh) {
                    const int8_t *row = row_ptr + kh * paddedW * inC;
                    for (int kw = 0; kw < poolSize; ++kw) {
                        const int8_t *ptr = row + kw * inC;
                        vint8m8_t vin = __riscv_vle8_v_i8m8(ptr, vl);
                        vmaxv = __riscv_vmax_vv_i8m8(vmaxv, vin, vl);
                    }
                }
                __riscv_vse8_v_i8m8(&out[(oh * outW + ow) * inC + c], vmaxv, vl);
                c += (int)vl;
            }
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

void maxpool2d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);

    const float *in = (const float *)padded_input;
    float *out = (float *)output;
    for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
        for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e32m8(inC - c);
                vfloat32m8_t vmaxv = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);
                for (int kh = 0; kh < poolSize; ++kh) {
                    int h = h_start + kh;
                    for (int kw = 0; kw < poolSize; ++kw) {
                        int w = w_start + kw;
                        const float *ptr = &in[(h * paddedW + w) * inC + c];
                        vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                        vmaxv = __riscv_vfmax_vv_f32m8(vmaxv, vin, vl);
                    }
                }
                __riscv_vse32_v_f32m8(&out[(oh * outW + ow) * inC + c], vmaxv, vl);
                c += (int)vl;
            }
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.

void avgpool2d_int8_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    int window_elems = poolSize * poolSize;

    const int8_t *in = (const int8_t *)padded_input;
    int8_t *out = (int8_t *)output;
    for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
        for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e8m4(inC - c);
                vint16m8_t vsum = __riscv_vmv_v_x_i16m8(0, vl);
                const int8_t *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                for (int kh = 0; kh < poolSize; ++kh) {
                    const int8_t *row = row_ptr + kh * paddedW * inC;
                    for (int kw = 0; kw < poolSize; ++kw) {
                        const int8_t *ptr = row + kw * inC;
                        vint8m4_t vin8 = __riscv_vle8_v_i8m4(ptr, vl);
                        vint16m8_t vin16 = __riscv_vsext_vf2_i16m8(vin8, vl);
                        vsum = __riscv_vadd_vv_i16m8(vsum, vin16, vl);
                    }
                }

                if (window_elems > 0) {
                    vsum = __riscv_vdiv_vx_i16m8(vsum, window_elems, vl);
                }
                vsum = __riscv_vmax_vx_i16m8(vsum, INT8_MIN, vl);
                vsum = __riscv_vmin_vx_i16m8(vsum, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum, vl);
                __riscv_vse8_v_i8m4(&out[(oh * outW + ow) * inC + c], vout, vl);
                c += (int)vl;
            }
        }
    }
    
    if (padding > 0)
        safe_free(padded_input);
}

void avgpool2d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;
    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    int window_elems = poolSize * poolSize;

    const float *in = (const float *)padded_input;
    float *out = (float *)output;
    for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
        for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e32m8(inC - c);
                vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

                const float *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                for (int kh = 0; kh < poolSize; ++kh) {
                    const float *row = row_ptr + kh * paddedW * inC;
                    for (int kw = 0; kw < poolSize; ++kw) {
                        const float *ptr = row + kw * inC;
                        vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                        vsum = __riscv_vfadd_vv_f32m8(vsum, vin, vl);
                    }
                }

                if (window_elems > 0) {
                    float inv = 1.0f / (float)window_elems;
                    vsum = __riscv_vfmul_vf_f32m8(vsum, inv, vl);
                }

                __riscv_vse32_v_f32m8(&out[(oh * outW + ow) * inC + c], vsum, vl);
                c += (int)vl;
            }
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

void AdaptiveAvgPool2d_int8_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH  = layer->inputShape.H;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;

    const int8_t *in = (const int8_t *)input;
    int8_t *out = (int8_t *)output;
    for (int oh = 0; oh < outH; ++oh) {
        int h_start = (oh * inH) / outH;
        int h_end = ((oh + 1) * inH) / outH;
        for (int ow = 0; ow < outW; ++ow) {
            int w_start = (ow * inW) / outW;
            int w_end = ((ow + 1) * inW) / outW;
            for (int c = 0; c < inC; ++c) {
                int32_t sum = 0;
                int count = 0;
                for (int h = h_start; h < h_end; ++h) {
                    for (int w = w_start; w < w_end; ++w) {
                        sum += in[(h * inW + w) * inC + c];
                        count++;
                    }
                }
                out[(oh * outW + ow) * inC + c] = (count > 0) ? (int8_t)(sum / count) : 0;
            }
        }
    }
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

void AdaptiveAvgPool2d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH  = layer->inputShape.H;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;

    const float *in = (const float *)input;
    float *out = (float *)output;

    for (int oh = 0; oh < outH; ++oh) {
        int h_start = (oh * inH) / outH;
        int h_end = ((oh + 1) * inH) / outH;
        for (int ow = 0; ow < outW; ++ow) {
            int w_start = (ow * inW) / outW;
            int w_end = ((ow + 1) * inW) / outW;
            for (int c = 0; c < inC; ++c) {
                float sum = 0.0f;
                int count = 0;
                for (int h = h_start; h < h_end; ++h) {
                    for (int w = w_start; w < w_end; ++w) {
                        sum += in[(h * inW + w) * inC + c];
                        count++;
                    }
                }
                out[(oh * outW + ow) * inC + c] = (count > 0) ? sum / count : 0.0f;
            }
        }
    }
}
