#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_tensor_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void transpose_vpu(NNModule *layer, void *input, void *output)
{
    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    TransposeType mode = layer->params.transpose.mode;

    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float *)input;
        float *out = (float *)output;
        switch (mode) {
        case TRANSPOSE_CW_TO_WC:
            for (int c = 0; c < channels; ++c) {
                const float *in_row = &in[c * width];
                float *out_col = &out[c];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e32m8(width - w);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(&in_row[w], vl);
                    __riscv_vsse32_v_f32m8(
                        &out_col[w * channels],
                        (ptrdiff_t)(channels * sizeof(float)),
                        v,
                        vl);
                    w += (int)vl;
                }
            }
            break;
        case TRANSPOSE_WC_TO_CW:
            for (int c = 0; c < channels; ++c) {
                const float *in_col = &in[c];
                float *out_row = &out[c * width];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e32m8(width - w);
                    vfloat32m8_t v = __riscv_vlse32_v_f32m8(
                        &in_col[w * channels],
                        (ptrdiff_t)(channels * sizeof(float)),
                        vl);
                    __riscv_vse32_v_f32m8(&out_row[w], v, vl);
                    w += (int)vl;
                }
            }
            break;
        default:
            printf("Unsupported transpose mode in transpose_vpu\n");
            exit(EXIT_FAILURE);
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t *)input;
        int8_t *out = (int8_t *)output;
        switch (mode) {
        case TRANSPOSE_CW_TO_WC:
            for (int c = 0; c < channels; ++c) {
                const int8_t *in_row = &in[c * width];
                int8_t *out_col = &out[c];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e8m8(width - w);
                    vint8m8_t v = __riscv_vle8_v_i8m8(&in_row[w], vl);
                    __riscv_vsse8_v_i8m8(
                        &out_col[w * channels],
                        (ptrdiff_t)channels,
                        v,
                        vl);
                    w += (int)vl;
                }
            }
            break;
        case TRANSPOSE_WC_TO_CW:
            for (int c = 0; c < channels; ++c) {
                const int8_t *in_col = &in[c];
                int8_t *out_row = &out[c * width];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e8m8(width - w);
                    vint8m8_t v = __riscv_vlse8_v_i8m8(
                        &in_col[w * channels],
                        (ptrdiff_t)channels,
                        vl);
                    __riscv_vse8_v_i8m8(&out_row[w], v, vl);
                    w += (int)vl;
                }
            }
            break;
        default:
            printf("Unsupported transpose mode in transpose_vpu\n");
            exit(EXIT_FAILURE);
        }
        break;
    }
    default:
        printf("Unsupported dtype in transpose_vpu\n");
        exit(EXIT_FAILURE);
    }
}

void save_vpu(NNModule *layer, void *input, void *output)
{
    // Input: arbitrary tensor in NHWC/1D (no change), copied into params.save.buffer.
    if (!layer->params.save.buffer || layer->params.save.bytes == 0) {
        printf("Save layer error: buffer or bytes not initialized\n");
        exit(EXIT_FAILURE);
    }

    size_t bytes = layer->params.save.bytes;
    int8_t *src_ptr = (int8_t *)input;
    int8_t *skip_ptr = (int8_t *)layer->params.save.buffer;
    (void)output; // output is intentionally untouched to avoid CPU writes

    for (size_t remaining = bytes; remaining > 0; ) {
        size_t vl = __riscv_vsetvl_e8m8(remaining);
        vint8m8_t v = __riscv_vle8_v_i8m8(src_ptr, vl);
        __riscv_vse8_v_i8m8(skip_ptr, v, vl);
        src_ptr += vl;
        skip_ptr += vl;
        remaining -= vl;
    }
}

void add_vpu(NNModule *layer, void *input, void *output)
{
    // Input: main tensor and skip tensor (same shape), both NHWC/1D; Output: same layout.
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
        activate_store_chunk_kernel_f32m8_t activation_kernel =
            select_activate_store_chunk_kernel_f32m8(layer->activation);
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t v0 = __riscv_vle32_v_f32m8(&in[i], vl);
            vfloat32m8_t v1 = __riscv_vle32_v_f32m8(&skip[i], vl);
            vfloat32m8_t vsum = __riscv_vfadd_vv_f32m8(v0, v1, vl);
            activation_kernel(vsum, &out[i], vl);
            i += (int)vl;
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *skip = (int8_t *)layer->params.add.skip;
        int8_t *out = (int8_t *)output;
        int len = (int)(layer->params.add.bytes / sizeof(int8_t));
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e8m4(len - i);
            vint8m4_t v0 = __riscv_vle8_v_i8m4(&in[i], vl);
            vint8m4_t v1 = __riscv_vle8_v_i8m4(&skip[i], vl);
            vint16m8_t v0_16 = __riscv_vsext_vf2_i16m8(v0, vl);
            vint16m8_t v1_16 = __riscv_vsext_vf2_i16m8(v1, vl);
            vint16m8_t vsum16 = __riscv_vadd_vv_i16m8(v0_16, v1_16, vl);

            if (layer->activation == NONE) {
                // clamp to int8 then narrow back
                vsum16 = __riscv_vmax_vx_i16m8(vsum16, INT8_MIN, vl);
                vsum16 = __riscv_vmin_vx_i16m8(vsum16, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum16, vl);
                __riscv_vse8_v_i8m4(&out[i], vout, vl);
            } else {
                // store to temp buffer for scalar activation
                // (activation_i8 is scalar; keep old behavior)
                vsum16 = __riscv_vmax_vx_i16m8(vsum16, INT8_MIN, vl);
                vsum16 = __riscv_vmin_vx_i16m8(vsum16, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum16, vl);
                __riscv_vse8_v_i8m4(&out[i], vout, vl);
                for (size_t t = 0; t < vl; ++t) {
                    out[i + t] = activate_i8(out[i + t], layer->activation);
                }
            }
            i += (int)vl;
        }
        break;
    }
    default:
        printf("Unsupported dtype in add_vpu\n");
        exit(EXIT_FAILURE);
    }
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/
