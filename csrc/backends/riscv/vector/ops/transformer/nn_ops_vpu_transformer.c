#include "nn_ops_vpu_transformer_internal.h"

#include "nn_utils.h"
#include <math.h>
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void linear_rows_fp32_rvv(const float *input,
                                 int rows,
                                 int in_dim,
                                 int out_dim,
                                 const float *weights_transposed,
                                 const float *bias,
                                 float *output)
{
    for (int r = 0; r < rows; ++r) {
        const float *row = &input[r * in_dim];
        float *dst = &output[r * out_dim];
        for (int o = 0; o < out_dim; ) {
            size_t vl = __riscv_vsetvl_e32m8(out_dim - o);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias[o], vl);
            for (int i = 0; i < in_dim; ++i) {
                float input_value = row[i];
                if (input_value == 0.0f)
                    continue;
                const float *weights = &weights_transposed[i * out_dim + o];
                vfloat32m8_t vweights = __riscv_vle32_v_f32m8(weights, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, input_value, vweights, vl);
            }
            __riscv_vse32_v_f32m8(&dst[o], vacc, vl);
            o += (int)vl;
        }
    }
}

void attention1d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    if (layer->dtype != ELEM_FLOAT32) {
        printf("Unsupported dtype in attention1d_fp32_vpu\n");
        exit(EXIT_FAILURE);
    }

    int seq_len = layer->inputShape.W;
    int embed_dim = layer->inputShape.C;
    int qkv_dim = embed_dim * 3;
    int num_heads = layer->params.attention.num_heads;
    int head_dim = layer->params.attention.head_dim;
    float scale = layer->params.attention.scale;
    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *input_weights = (const float *)layer->params.attention.in_proj_weight_rvv;
    const float *input_bias = (const float *)layer->params.attention.in_proj_bias;
    const float *output_weights = (const float *)layer->params.attention.out_proj_weight_rvv;
    const float *output_bias = (const float *)layer->params.attention.out_proj_bias;
    float *qkv = (float *)layer->params.attention.qkv_buffer;
    float *context = (float *)layer->params.attention.ctx_buffer;
    float *projection = (float *)layer->params.attention.proj_buffer;
    float *scores = (float *)layer->params.attention.score_buffer;

    linear_rows_fp32_rvv(input_f32, seq_len, embed_dim, qkv_dim, input_weights, input_bias, qkv);
    memset(context, 0, (size_t)seq_len * embed_dim * sizeof(float));

    for (int t = 0; t < seq_len; ++t) {
        const float *query_row = &qkv[t * qkv_dim];
        for (int h = 0; h < num_heads; ++h) {
            const float *query = &query_row[h * head_dim];
            for (int s = 0; s < seq_len; ++s) {
                const float *source = &qkv[s * qkv_dim];
                const float *key = &source[embed_dim + h * head_dim];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ) {
                    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);
                    vfloat32m8_t queries = __riscv_vle32_v_f32m8(&query[d], vl);
                    vfloat32m8_t keys = __riscv_vle32_v_f32m8(&key[d], vl);
                    vfloat32m8_t products = __riscv_vfmul_vv_f32m8(queries, keys, vl);
                    vfloat32m1_t init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
                    vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m8_f32m1(products, init, vl);
                    dot += __riscv_vfmv_f_s_f32m1_f32(reduced);
                    d += (int)vl;
                }
                scores[s] = dot * scale;
            }
            softmax_f32(scores, scores, seq_len);

            float *context_head = &context[t * embed_dim + h * head_dim];
            for (int s = 0; s < seq_len; ++s) {
                float coefficient = scores[s];
                const float *source = &qkv[s * qkv_dim];
                const float *value = &source[2 * embed_dim + h * head_dim];
                for (int d = 0; d < head_dim; ) {
                    size_t vl = __riscv_vsetvl_e32m8(head_dim - d);
                    vfloat32m8_t accumulated = __riscv_vle32_v_f32m8(&context_head[d], vl);
                    vfloat32m8_t values = __riscv_vle32_v_f32m8(&value[d], vl);
                    accumulated = __riscv_vfmacc_vf_f32m8(accumulated, coefficient, values, vl);
                    __riscv_vse32_v_f32m8(&context_head[d], accumulated, vl);
                    d += (int)vl;
                }
            }
        }
    }

    linear_rows_fp32_rvv(context, seq_len, embed_dim, embed_dim,
                         output_weights, output_bias, projection);
    memcpy(output_f32, projection, (size_t)seq_len * embed_dim * sizeof(float));
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/
