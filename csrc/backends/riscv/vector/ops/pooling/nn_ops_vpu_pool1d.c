#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_pool1d_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void maxpool1d_int8_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;

    const int8_t *in = (const int8_t*)input;
    int8_t *out = (int8_t*)output;

    for (int pos = 0; pos < outW; pos++) {
        int start = pos * stride;
        int end = start + poolSize;
        if (end > inW) end = inW;

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e8m8(inC - c);
            vint8m8_t vmaxv = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);
            for (int w = start; w < end; w++) {
                const int8_t *ptr = &in[w * inC + c];
                vint8m8_t vin = __riscv_vle8_v_i8m8(ptr, vl);
                vmaxv = __riscv_vmax_vv_i8m8(vmaxv, vin, vl);
            }
            __riscv_vse8_v_i8m8(&out[pos * inC + c], vmaxv, vl);
            c += (int)vl;
        }
    }
}

void maxpool1d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;

    const float *in = (const float*)input;
    float *out = (float*)output;

    for (int pos = 0; pos < outW; pos++) {
        int start = pos * stride;
        int end = start + poolSize;
        if (end > inW) end = inW;

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e32m8(inC - c);
            vfloat32m8_t vmaxv = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);
            for (int w = start; w < end; w++) {
                const float *ptr = &in[w * inC + c];
                vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                vmaxv = __riscv_vfmax_vv_f32m8(vmaxv, vin, vl);
            }
            __riscv_vse32_v_f32m8(&out[pos * inC + c], vmaxv, vl);
            c += (int)vl;
        }
    } 
}

// Input: NHWC with H=1, layout [N=1][W][C]; Output: same layout, W pooled.

void avgpool1d_int8_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;

    const int8_t *in = (const int8_t*)input;
    int8_t *out = (int8_t*)output;

    for (int pos = 0; pos < outW; pos++) {
        int start = pos * stride;
        int end = start + poolSize;
        if (end > inW) end = inW;
        int len = end - start;

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e8m4(inC - c);
            vint16m8_t vsum = __riscv_vmv_v_x_i16m8(0, vl);
            for (int w = start; w < end; w++) {
                const int8_t *ptr = &in[w * inC + c];
                vint8m4_t vin8 = __riscv_vle8_v_i8m4(ptr, vl);
                vint16m8_t vin16 = __riscv_vsext_vf2_i16m8(vin8, vl);
                vsum = __riscv_vadd_vv_i16m8(vsum, vin16, vl);
            }
            if (len > 0)
                vsum = __riscv_vdiv_vx_i16m8(vsum, len, vl);
            vsum = __riscv_vmax_vx_i16m8(vsum, INT8_MIN, vl);
            vsum = __riscv_vmin_vx_i16m8(vsum, INT8_MAX, vl);
            vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum, vl);
            __riscv_vse8_v_i8m4(&out[pos * inC + c], vout, vl);
            c += (int)vl;
        }
    }
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

void avgpool1d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;

    const float *in = (const float*)input;
    float *out = (float*)output;

    for (int pos = 0; pos < outW; pos++) {
        int start = pos * stride;
        int end = start + poolSize;
        if (end > inW) end = inW;
        int len = end - start;

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e32m8(inC - c);
            vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            for (int w = start; w < end; w++) {
                const float *ptr = &in[w * inC + c];
                vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                vsum = __riscv_vfadd_vv_f32m8(vsum, vin, vl);
            }
            float inv = len > 0 ? 1.0f / (float)len : 0.0f;
            vsum = __riscv_vfmul_vf_f32m8(vsum, inv, vl);
            __riscv_vse32_v_f32m8(&out[pos * inC + c], vsum, vl);
            c += (int)vl;
        }
    }
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.

void AdaptiveMaxPool1d_wc_int8_vpu(NNModule *layer, void *input, void *output)
{
    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int outW = layer->outputShape.W;

    if (outW == 1) {
        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e8m8(inC - c);
            vint8m8_t vmax = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);

            for (int w = 0; w < inW; ++w) {
                const int8_t *in_ptr = &input_i8[w * inC + c];
                vint8m8_t vin = __riscv_vle8_v_i8m8(in_ptr, vl);
                vmax = __riscv_vmax_vv_i8m8(vmax, vin, vl);
            }

            __riscv_vse8_v_i8m8(&output_i8[c], vmax, vl);
            c += (int)vl;
        }
        return;
    }

    // Input/Output: N=1,H=1; layout [W][C]. Output W pooled.
    for (int pos = 0; pos < outW; ++pos) {
        int start = (pos * inW) / outW;
        int end = ((pos + 1) * inW) / outW; // floor (aligned with legacy path)
        if (end > inW)
            end = inW;

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e8m8(inC - c);
            vint8m8_t vmax = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);

            for (int w = start; w < end; ++w) {
                const int8_t *in_ptr = &input_i8[w * inC + c];
                vint8m8_t vin = __riscv_vle8_v_i8m8(in_ptr, vl);
                vmax = __riscv_vmax_vv_i8m8(vmax, vin, vl);
            }

            __riscv_vse8_v_i8m8(&output_i8[pos * inC + c], vmax, vl);
            c += (int)vl;
        }
    }
}

void AdaptiveMaxPool1d_wc_fp32_vpu(NNModule *layer, void *input, void *output)
{
    const float *input_f32 = (const float*)input;
    float *output_f32 = (float*)output;
    int inW= layer->inputShape.W;
    int inC= layer->inputShape.C;
    int outW= layer->outputShape.W;
    
    // Input/Output: N=1,H=1; layout [W][C]. Output W pooled.
    for (int pos = 0; pos < outW; ++pos) {
        int start = (pos * inW) / outW;
        int end = ((pos + 1) * inW) / outW;
        if (end > inW) {
            end = inW;
        }

        for (int c = 0; c < inC; ) {
            size_t vl = __riscv_vsetvl_e32m8(inC - c);
            vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);

            for (int w = start; w < end; ++w) {
                const float *in_ptr = &input_f32[w * inC + c];
                vfloat32m8_t vin = __riscv_vle32_v_f32m8(in_ptr, vl);
                vmax = __riscv_vfmax_vv_f32m8(vmax, vin, vl);
            }

            __riscv_vse32_v_f32m8(&output_f32[pos * inC + c], vmax, vl);
            c += (int)vl;
        }
    }
}
