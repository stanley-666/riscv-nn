/* SPDX-FileContributor: Person: Stanley Lee */
#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_conv2d_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void conv2d_int8_vpu_m8(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    int paddedW = inW + 2 * padding;

    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv2d_vpu.\n");
        exit(EXIT_FAILURE);
    }

    const int8_t *input_i8 = (const int8_t *)padded_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv2d.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv2d.weights_rvv;
    const float *M = (const float *)layer->params.conv2d.M;
    const int32_t *Z = (const int32_t *)layer->params.conv2d.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    if (!weight_buffer) {
        printf("Error: conv2d weights_rvv not initialized.\n");
        exit(EXIT_FAILURE);
    }

    for (int oh = 0, base_h = 0; oh < outH; ++oh, base_h += stride) {
        for (int ow = 0, base_w = 0; ow < outW; ++ow, base_w += stride) {
            const int8_t *base_in = &input_i8[(base_h * paddedW + base_w) * inC];
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e16m4(outC - oc);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int kh = 0; kh < filterSize; ++kh) {
                    const int8_t *row_in = base_in + kh * paddedW * inC;
                    int col_base = kh * filterSize * inC;
                    for (int kw = 0; kw < filterSize; ++kw) {
                        const int8_t *in_ptr = row_in + kw * inC;
                        int col_idx = col_base + kw * inC;
                        const int16_t *wt_base = &weight_buffer[col_idx * outC + oc];
                        for (int ic = 0; ic < inC; ++ic) {
                            int8_t inval = in_ptr[ic];
                            if (inval == 0)
                                continue;
                            vint16m4_t vrow16 = __riscv_vmv_v_x_i16m4(inval, vl);
                            const int16_t *wt = wt_base + ic * outC;
                            vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                            vacc = __riscv_vwmacc_vv_i32m8(vacc, vrow16, vwt16, vl);
                        }
                    }
                }
                vint32m8_t vbias = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
                vacc = __riscv_vadd_vv_i32m8(vacc, vbias, vl);

                rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[(oh * outW + ow) * outC + oc], vl);
                oc += vl;
            }
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

void conv2d_int8_vpu_im2col_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_2d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv2d_int8_vpu_im2col_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv2d.filterSize * layer->params.conv2d.filterSize *
               layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv2d.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv2d.weights_rvv;
    const float *M = (const float *)layer->params.conv2d.M;
    const int32_t *Z = (const int32_t *)layer->params.conv2d.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int out_idx = 0; out_idx < outH * outW; ++out_idx) {
        const int8_t *input_row = &input_i8[out_idx * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int8_t inval = input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[out_idx * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.

void conv2d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    int paddedW = inW + 2 * padding;

    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv2d_vpu.\n");
        exit(EXIT_FAILURE);
    }

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv2d.bias;
    const float *weight_buffer = (const float *)layer->params.conv2d.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int oh = 0, base_h = 0; oh < outH; ++oh, base_h += stride) {
        for (int ow = 0, base_w = 0; ow < outW; ++ow, base_w += stride) {
            const float *base_in = &input_f32[(base_h * paddedW + base_w) * inC];
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e32m8(outC - oc);
                vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

                for (int kh = 0; kh < filterSize; ++kh) {
                    const float *row_in = base_in + kh * paddedW * inC;
                    int col_base = kh * filterSize * inC;
                    for (int kw = 0; kw < filterSize; ++kw) {
                        const float *in_ptr = row_in + kw * inC;
                        int col_idx = col_base + kw * inC;
                        const float *wt_base = &weight_buffer[col_idx * outC + oc];
                        for (int ic = 0; ic < inC; ++ic) {
                            float inval = in_ptr[ic];
                            const float *wt = wt_base + ic * outC;
                            vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                            vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                        }
                    }
                }

                activation_kernel(vacc, &output_f32[(oh * outW + ow) * outC + oc], vl);
                oc += vl;
            }
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

void conv2d_fp32_vpu_im2col(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_2d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv2d_fp32_vpu_im2col.\n");
        exit(EXIT_FAILURE);
    }

    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv2d.filterSize * layer->params.conv2d.filterSize *
               layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv2d.bias;
    const float *weight_buffer = (const float *)layer->params.conv2d.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int out_idx = 0; out_idx < outH * outW; ++out_idx) {
        const float *input_row = &input_f32[out_idx * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f)
                    continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[out_idx * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

// Input: NHWC with H=1, layout [N=1][W][C]; Output: same layout, W pooled.

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/
