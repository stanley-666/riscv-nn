/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "nn_layer.h"
#include "backends/riscv/vector/ops/conv1d/nn_ops_vpu_conv1d_fp32_internal.h"
#include "backends/riscv/vector/ops/fully_connected/nn_ops_vpu_fc_internal.h"
#include "backends/riscv/vector/ops/pooling/nn_ops_vpu_pool1d_internal.h"
#include "backends/riscv/vector/ops/tensor/nn_ops_vpu_tensor_internal.h"
#include "backends/riscv/vector/ops/transformer/nn_ops_vpu_transformer_internal.h"
#include "backends/riscv/vector/ops/normalization/nn_ops_vpu_normalization_internal.h"
#include "nn_utils.h"

#include "kyber_nouv_sample.h"
#include "kyber_nouv_weights.h"
#include "kyber_nouv_runner.h"

#define KYBER_MAX_ACT_ELEMS (KYBER_NOUV_STAGE3_LEN * KYBER_NOUV_EMBED_DIM)
#define KYBER_ATTN_SKIP_ELEMS (KYBER_NOUV_TRACE_SEQ_LEN * KYBER_NOUV_EMBED_DIM)

static float act_a[KYBER_MAX_ACT_ELEMS] __attribute__((aligned(64)));
static float act_b[KYBER_MAX_ACT_ELEMS] __attribute__((aligned(64)));
static float skip_buf[KYBER_ATTN_SKIP_ELEMS] __attribute__((aligned(64)));
static float pooled_avg[KYBER_NOUV_EMBED_DIM] __attribute__((aligned(64)));
static float pooled_max[KYBER_NOUV_EMBED_DIM] __attribute__((aligned(64)));
static float fused_vec[KYBER_NOUV_FUSED_DIM] __attribute__((aligned(64)));
static float head_hidden[KYBER_NOUV_HEAD_HIDDEN_DIM] __attribute__((aligned(64)));
static float logits[KYBER_NOUV_NUM_CLASSES] __attribute__((aligned(64)));

static uint64_t read_rdcycle(void)
{
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static void print_top5(const float *values, int len)
{
    int top_idx[KYBER_NOUV_SAMPLE_TOPK];
    float top_val[KYBER_NOUV_SAMPLE_TOPK];
    for (int i = 0; i < KYBER_NOUV_SAMPLE_TOPK; ++i) {
        top_idx[i] = -1;
        top_val[i] = -FLT_MAX;
    }

    for (int i = 0; i < len; ++i) {
        float v = values[i];
        for (int k = 0; k < KYBER_NOUV_SAMPLE_TOPK; ++k) {
            if (v > top_val[k]) {
                for (int shift = KYBER_NOUV_SAMPLE_TOPK - 1; shift > k; --shift) {
                    top_val[shift] = top_val[shift - 1];
                    top_idx[shift] = top_idx[shift - 1];
                }
                top_val[k] = v;
                top_idx[k] = i;
                break;
            }
        }
    }

    printf("Top-5 logits:\n");
    for (int k = 0; k < KYBER_NOUV_SAMPLE_TOPK; ++k) {
        printf("  [%d] class=%d logit=%.6f", k, top_idx[k], top_val[k]);
        if (top_idx[k] == (int)KYBER_NOUV_SAMPLE_TOP5_CLASS[k]) {
            printf("  (matches expected rank %d)", k);
        }
        printf("\n");
    }
}

void kyber_nouv_run(void)
{
    CNN *model = createCNN();

    NNModule *transpose = nn_Transpose(KYBER_NOUV_TRACE_LEN, KYBER_NOUV_TRACE_IN_CHANNELS, TRANSPOSE_CW_TO_WC, ELEM_FLOAT32);
    NNModule *conv1 = nn_Conv1d(KYBER_NOUV_TRACE_IN_CHANNELS, KYBER_NOUV_TRACE_LEN, 9, 48, 2, 4, GELU, kyber_conv1_weight, kyber_conv1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv2 = nn_Conv1d(48, KYBER_NOUV_STAGE1_LEN, 7, 96, 2, 3, GELU, kyber_conv2_weight, kyber_conv2_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv3 = nn_Conv1d(96, KYBER_NOUV_STAGE2_LEN, 5, 192, 2, 2, GELU, kyber_conv3_weight, kyber_conv3_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv4 = nn_Conv1d(192, KYBER_NOUV_STAGE3_LEN, 5, 192, 2, 2, GELU, kyber_conv4_weight, kyber_conv4_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv5 = nn_Conv1d(192, KYBER_NOUV_STAGE4_LEN, 5, 192, 1, 2, GELU, kyber_conv5_weight, kyber_conv5_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *save = nn_Save(KYBER_NOUV_EMBED_DIM, 1, KYBER_NOUV_TRACE_SEQ_LEN, ELEM_FLOAT32);
    NNModule *attn = nn_MultiHeadAttention1d(
        KYBER_NOUV_TRACE_SEQ_LEN,
        KYBER_NOUV_EMBED_DIM,
        KYBER_NOUV_ATTN_HEADS,
        kyber_attn_in_proj_weight,
        kyber_attn_in_proj_bias,
        kyber_attn_out_proj_weight,
        kyber_attn_out_proj_bias,
        ELEM_FLOAT32);
    NNModule *add = nn_Add(KYBER_NOUV_EMBED_DIM, 1, KYBER_NOUV_TRACE_SEQ_LEN, NONE, skip_buf, ELEM_FLOAT32);
    NNModule *attn_norm = nn_LayerNorm1d(KYBER_NOUV_TRACE_SEQ_LEN, KYBER_NOUV_EMBED_DIM, kyber_attn_norm_weight, kyber_attn_norm_bias, 1e-5f, ELEM_FLOAT32);
    NNModule *avgpool = nn_Pool1d(KYBER_NOUV_EMBED_DIM, KYBER_NOUV_TRACE_SEQ_LEN, KYBER_NOUV_TRACE_SEQ_LEN, 1, AVG_POOL, ELEM_FLOAT32);
    NNModule *maxpool = nn_AdaptiveMaxPool1d(KYBER_NOUV_EMBED_DIM, KYBER_NOUV_TRACE_SEQ_LEN, 1, ELEM_FLOAT32);
    NNModule *head_norm = nn_LayerNorm1d(1, KYBER_NOUV_FUSED_DIM, kyber_head_ln_weight, kyber_head_ln_bias, 1e-5f, ELEM_FLOAT32);
    NNModule *head_fc1 = nn_Linear(KYBER_NOUV_FUSED_DIM, KYBER_NOUV_HEAD_HIDDEN_DIM, GELU, kyber_head_fc1_weight, kyber_head_fc1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *head_fc2 = nn_Linear(KYBER_NOUV_HEAD_HIDDEN_DIM, KYBER_NOUV_NUM_CLASSES, NONE, kyber_head_fc2_weight, kyber_head_fc2_bias, NULL, NULL, ELEM_FLOAT32);

    save->params.save.buffer = skip_buf;
    save->params.save.bytes = sizeof(skip_buf);

    addLayer(model, transpose);
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, conv4);
    addLayer(model, conv5);
    addLayer(model, save);
    addLayer(model, attn);
    addLayer(model, add);
    addLayer(model, attn_norm);
    addLayer(model, avgpool);
    addLayer(model, maxpool);
    addLayer(model, head_norm);
    addLayer(model, head_fc1);
    addLayer(model, head_fc2);

    uint64_t start_cycle = read_rdcycle();
    transpose_vpu(transpose, (void *)KYBER_NOUV_SAMPLE_TRACE_X, act_a);
    conv1d_fp32_vpu_im2col_unroll8_acc2_m8(conv1, act_a, act_b);
    conv1d_fp32_vpu_im2col_unroll8_acc2_m8(conv2, act_b, act_a);
    conv1d_fp32_vpu_im2col_unroll8_acc2_m8(conv3, act_a, act_b);
    conv1d_fp32_vpu_im2col_unroll8_acc2_m8(conv4, act_b, act_a);
    conv1d_fp32_vpu_im2col_unroll8_acc2_m8(conv5, act_a, act_b);
    save_vpu(save, act_b, act_a);
    attention1d_fp32_vpu(attn, act_b, act_a);
    add_vpu(add, act_a, act_b);
    layernorm1d_fp32_vpu(attn_norm, act_b, act_a);
    avgpool1d_fp32_vpu(avgpool, act_a, pooled_avg);
    AdaptiveMaxPool1d_wc_fp32_vpu(maxpool, act_a, pooled_max);

    memcpy(&fused_vec[0], pooled_avg, sizeof(pooled_avg));
    memcpy(&fused_vec[KYBER_NOUV_EMBED_DIM], pooled_max, sizeof(pooled_max));
    fused_vec[KYBER_NOUV_FUSED_DIM - 1] = KYBER_NOUV_SAMPLE_BYTE_POS;

    layernorm1d_fp32_vpu(head_norm, fused_vec, fused_vec);
    fullyconnected_fp32_vpu(head_fc1, fused_vec, head_hidden);
    fullyconnected_fp32_vpu(head_fc2, head_hidden, logits);
    uint64_t end_cycle = read_rdcycle();

    int argmax = 0;
    for (int i = 1; i < KYBER_NOUV_NUM_CLASSES; ++i) {
        if (logits[i] > logits[argmax]) {
            argmax = i;
        }
    }

    printf("Kyber nouv RVV inference\n");
    printf("message_idx=%d byte_idx=%d\n", KYBER_NOUV_SAMPLE_MESSAGE_IDX, KYBER_NOUV_SAMPLE_BYTE_IDX);
    printf("cycles=%lu\n", end_cycle - start_cycle);
    printf("predicted_argmax=%d expected_argmax=%d target=%d\n",
           argmax,
           KYBER_NOUV_SAMPLE_PYTORCH_ARGMAX,
           KYBER_NOUV_SAMPLE_TARGET);
    print_top5(logits, KYBER_NOUV_NUM_CLASSES);

    freeCNN(model);
}

int main(void)
{
    kyber_nouv_run();
    return 0;
}
