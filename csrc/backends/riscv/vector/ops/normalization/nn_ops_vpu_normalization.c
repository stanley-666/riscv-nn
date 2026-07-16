/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */

#include "nn_ops_vpu_normalization_internal.h"

#include <math.h>
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>

void layernorm1d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    if (layer->dtype != ELEM_FLOAT32) {
        printf("Unsupported dtype in layernorm1d_fp32_vpu\n");
        exit(EXIT_FAILURE);
    }

    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *weight = (const float *)layer->params.layernorm.weight;
    const float *bias = (const float *)layer->params.layernorm.bias;
    float eps = layer->params.layernorm.eps;

    for (int w = 0; w < width; ++w) {
        const float *row = &input_f32[w * channels];
        float *dst = &output_f32[w * channels];
        double sum = 0.0;
        for (int c = 0; c < channels; ) {
            size_t vl = __riscv_vsetvl_e32m8(channels - c);
            vfloat32m8_t values = __riscv_vle32_v_f32m8(&row[c], vl);
            vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m8_f32m1(values, init, vl);
            sum += (double)__riscv_vfmv_f_s_f32m1_f32(reduced);
            c += (int)vl;
        }
        float mean = (float)(sum / (double)channels);

        double variance_sum = 0.0;
        for (int c = 0; c < channels; ) {
            size_t vl = __riscv_vsetvl_e32m8(channels - c);
            vfloat32m8_t values = __riscv_vle32_v_f32m8(&row[c], vl);
            values = __riscv_vfsub_vf_f32m8(values, mean, vl);
            values = __riscv_vfmul_vv_f32m8(values, values, vl);
            vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
            vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m8_f32m1(values, init, vl);
            variance_sum += (double)__riscv_vfmv_f_s_f32m1_f32(reduced);
            c += (int)vl;
        }
        float inv_std = 1.0f / sqrtf((float)(variance_sum / (double)channels) + eps);

        for (int c = 0; c < channels; ) {
            size_t vl = __riscv_vsetvl_e32m8(channels - c);
            vfloat32m8_t values = __riscv_vle32_v_f32m8(&row[c], vl);
            vfloat32m8_t weights = __riscv_vle32_v_f32m8(&weight[c], vl);
            vfloat32m8_t biases = __riscv_vle32_v_f32m8(&bias[c], vl);
            values = __riscv_vfsub_vf_f32m8(values, mean, vl);
            values = __riscv_vfmul_vf_f32m8(values, inv_std, vl);
            values = __riscv_vfmul_vv_f32m8(values, weights, vl);
            values = __riscv_vfadd_vv_f32m8(values, biases, vl);
            __riscv_vse32_v_f32m8(&dst[c], values, vl);
            c += (int)vl;
        }
    }
}
