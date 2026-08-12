/* SPDX-FileContributor: Person: Stanley Lee */
/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_transformer_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

void attention1d_cpu(NNModule *layer, void *input, void *output)
{
    if (layer->dtype != ELEM_FLOAT32) {
        printf("Unsupported dtype in attention1d_cpu\n");
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
    float *qkv = (float *)layer->params.attention.qkv_buffer;
    float *ctx = (float *)layer->params.attention.ctx_buffer;
    float *proj = (float *)layer->params.attention.proj_buffer;
    float *scores = (float *)layer->params.attention.score_buffer;
    const float *in_w = (const float *)layer->params.attention.in_proj_weight;
    const float *in_b = (const float *)layer->params.attention.in_proj_bias;
    const float *out_w = (const float *)layer->params.attention.out_proj_weight;
    const float *out_b = (const float *)layer->params.attention.out_proj_bias;

    for (int t = 0; t < seq_len; ++t) {
        const float *row = &input_f32[t * embed_dim];
        float *dst = &qkv[t * qkv_dim];
        for (int o = 0; o < qkv_dim; ++o) {
            float sum = in_b[o];
            for (int i = 0; i < embed_dim; ++i)
                sum += row[i] * in_w[o * embed_dim + i];
            dst[o] = sum;
        }
    }

    memset(ctx, 0, (size_t)seq_len * embed_dim * sizeof(float));
    for (int t = 0; t < seq_len; ++t) {
        const float *q_row = &qkv[t * qkv_dim];
        for (int h = 0; h < num_heads; ++h) {
            const float *q = &q_row[h * head_dim];
            for (int s = 0; s < seq_len; ++s) {
                const float *src = &qkv[s * qkv_dim];
                const float *k = &src[embed_dim + h * head_dim];
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += q[d] * k[d];
                scores[s] = dot * scale;
            }
            softmax_f32(scores, scores, seq_len);

            float *ctx_head = &ctx[t * embed_dim + h * head_dim];
            for (int d = 0; d < head_dim; ++d)
                ctx_head[d] = 0.0f;
            for (int s = 0; s < seq_len; ++s) {
                const float *src = &qkv[s * qkv_dim];
                const float *v = &src[(2 * embed_dim) + h * head_dim];
                for (int d = 0; d < head_dim; ++d)
                    ctx_head[d] += scores[s] * v[d];
            }
        }
    }

    for (int t = 0; t < seq_len; ++t) {
        const float *row = &ctx[t * embed_dim];
        float *dst = &proj[t * embed_dim];
        for (int o = 0; o < embed_dim; ++o) {
            float sum = out_b[o];
            for (int i = 0; i < embed_dim; ++i)
                sum += row[i] * out_w[o * embed_dim + i];
            dst[o] = sum;
        }
    }

    memcpy(output_f32, proj, (size_t)seq_len * embed_dim * sizeof(float));
}


