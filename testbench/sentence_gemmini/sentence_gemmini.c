/*
 * Copyright 2026 Stanley Lee
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#if !defined(BAREMETAL) && !defined(SPIKE_PK)
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#include "random_embedding_gemmini.h"
#ifndef GEMMINI_ACCURACY_SWEEP
#define GEMMINI_ACCURACY_SWEEP 0
#endif

#if GEMMINI_ACCURACY_SWEEP
#include "sentence_dataset_gemmini.h"
#endif

#ifndef GEMMINI_USE_PER_TENSOR_WEIGHTS
#define GEMMINI_USE_PER_TENSOR_WEIGHTS 1
#endif

#if GEMMINI_USE_PER_TENSOR_WEIGHTS
#include "weights_q_gemmini_per_tensor.h"
#include "weights_gemmini_native_per_tensor.h"
#else
#include "weights_q_gemmini.h"
#include "weights_gemmini_native.h"
#endif

#define CONV1_OUT_SCALE 0.66297567f
#define CONV2_OUT_SCALE 2.63254571f
#define CONV3_OUT_SCALE 0.64812046f
#define FC1_OUT_SCALE 2.76002026f
#define FC2_OUT_SCALE 0.30590293f

#ifndef CONV1_REQUANT_SCALE
#define CONV1_REQUANT_SCALE 0.00358622f
#endif
#ifndef CONV2_REQUANT_SCALE
#define CONV2_REQUANT_SCALE 0.00005809f
#endif
#ifndef CONV3_REQUANT_SCALE
#define CONV3_REQUANT_SCALE 0.00734185f
#endif
#ifndef FC1_REQUANT_SCALE
#define FC1_REQUANT_SCALE 0.00000003f
#endif
#ifndef FC2_REQUANT_SCALE
#define FC2_REQUANT_SCALE 0.00297849f
#endif

#ifndef GEMMINI_PER_TENSOR_REQUANT
#define GEMMINI_PER_TENSOR_REQUANT 1
#endif

#ifndef GEMMINI_NATIVE_CONV
#define GEMMINI_NATIVE_CONV 1
#endif

#ifndef GEMMINI_STAGED_POOLING
#define GEMMINI_STAGED_POOLING GEMMINI_NATIVE_CONV
#endif

#ifndef GEMMINI_TIMING_SKIP_CPU_POOL
#define GEMMINI_TIMING_SKIP_CPU_POOL 0
#endif

#ifndef GEMMINI_FC_ON_CPU
#define GEMMINI_FC_ON_CPU 0
#endif

#define CONV1_IN_CHANNELS IN_CHANNELS
#define CONV1_OUT_CHANNELS 64
#define CONV1_KERNEL 5
#define CONV1_PADDING 2
#define CONV1_STRIDE 1
#define CONV1_OUT_COL_DIM ((IN_COL_DIM + 2 * CONV1_PADDING - CONV1_KERNEL) / CONV1_STRIDE + 1)

#define CONV2_IN_CHANNELS CONV1_OUT_CHANNELS
#define CONV2_OUT_CHANNELS 128
#define CONV2_KERNEL 5
#define CONV2_PADDING 2
#define CONV2_STRIDE 1
#define CONV2_OUT_COL_DIM ((CONV1_OUT_COL_DIM + 2 * CONV2_PADDING - CONV2_KERNEL) / CONV2_STRIDE + 1)

#define CONV3_IN_CHANNELS CONV2_OUT_CHANNELS
#define CONV3_OUT_CHANNELS 256
#define CONV3_KERNEL 3
#define CONV3_PADDING 1
#define CONV3_STRIDE 1
#define CONV3_OUT_COL_DIM ((CONV2_OUT_COL_DIM + 2 * CONV3_PADDING - CONV3_KERNEL) / CONV3_STRIDE + 1)

#define FC1_IN_FEATURES CONV3_OUT_CHANNELS
#define FC1_OUT_FEATURES 128
#define FC2_IN_FEATURES FC1_OUT_FEATURES
#define FC2_OUT_FEATURES 1

#define CONV1_K_DIM (CONV1_KERNEL * CONV1_IN_CHANNELS)
#define CONV2_K_DIM (CONV2_KERNEL * CONV2_IN_CHANNELS)
#define CONV3_K_DIM (CONV3_KERNEL * CONV3_IN_CHANNELS)
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#define MAX_CONV1D_TILE_ROWS 32
#define MAX_CONV1D_K_DIM MAX2(CONV1_K_DIM, MAX2(CONV2_K_DIM, CONV3_K_DIM))

#ifndef BAREMETAL_CPU_HZ
#define BAREMETAL_CPU_HZ 30000000UL
#endif

#define CONV1_OUT_BYTES ((size_t)BATCH_SIZE * IN_ROW_DIM * CONV1_OUT_COL_DIM * CONV1_OUT_CHANNELS * sizeof(elem_t))
#define CONV2_OUT_BYTES ((size_t)BATCH_SIZE * IN_ROW_DIM * CONV2_OUT_COL_DIM * CONV2_OUT_CHANNELS * sizeof(elem_t))
#define POOLED_BYTES    ((size_t)BATCH_SIZE * FC1_IN_FEATURES * sizeof(elem_t))
#define FC1_OUT_BYTES   ((size_t)BATCH_SIZE * FC1_OUT_FEATURES * sizeof(elem_t))
#define FC2_OUT_BYTES   ((size_t)BATCH_SIZE * FC2_OUT_FEATURES * sizeof(elem_t))

typedef struct {
    const char *name;
    uint64_t cycles;
    uint64_t compute_cycles;
    uint64_t requantize_cycles;
    uint64_t activation_cycles;
    uint64_t pool_cycles;
    size_t op_count;
    size_t bytes_in;
    size_t bytes_weights;
    size_t bytes_out;
} layer_profile_t;

enum {
    PROFILE_CONV1 = 0,
    PROFILE_CONV2,
    PROFILE_CONV3,
    PROFILE_POOL,
    PROFILE_FC1,
    PROFILE_FC2,
    PROFILE_COUNT
};

uint64_t read_rdcycle() {
    uint64_t cycle;
    __asm__ volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

static void init_profiles(layer_profile_t profiles[PROFILE_COUNT]) {
    profiles[PROFILE_CONV1] = (layer_profile_t){
        .name = "conv1",
        .op_count = 2ULL * BATCH_SIZE * CONV1_OUT_COL_DIM * CONV1_OUT_CHANNELS * CONV1_KERNEL * CONV1_IN_CHANNELS,
        .bytes_in = sizeof(input),
        .bytes_weights = sizeof(conv1_weight),
        .bytes_out = CONV1_OUT_BYTES,
    };
    profiles[PROFILE_CONV2] = (layer_profile_t){
        .name = "conv2",
        .op_count = 2ULL * BATCH_SIZE * CONV2_OUT_COL_DIM * CONV2_OUT_CHANNELS * CONV2_KERNEL * CONV2_IN_CHANNELS,
        .bytes_in = CONV1_OUT_BYTES,
        .bytes_weights = sizeof(conv2_weight),
        .bytes_out = CONV2_OUT_BYTES,
    };
    profiles[PROFILE_CONV3] = (layer_profile_t){
        .name = "conv3",
        .op_count = 2ULL * BATCH_SIZE * CONV3_OUT_COL_DIM * CONV3_OUT_CHANNELS * CONV3_KERNEL * CONV3_IN_CHANNELS,
        .bytes_in = CONV2_OUT_BYTES,
        .bytes_weights = sizeof(conv3_weight),
        .bytes_out = (size_t)BATCH_SIZE * CONV3_OUT_COL_DIM * CONV3_OUT_CHANNELS * sizeof(elem_t),
    };
    profiles[PROFILE_POOL] = (layer_profile_t){
        .name = "global_maxpool",
        .op_count = (size_t)(CONV3_OUT_COL_DIM - 1) * CONV3_OUT_CHANNELS,
        .bytes_in = (size_t)BATCH_SIZE * CONV3_OUT_COL_DIM * CONV3_OUT_CHANNELS * sizeof(elem_t),
        .bytes_weights = 0,
        .bytes_out = POOLED_BYTES,
    };
    profiles[PROFILE_FC1] = (layer_profile_t){
        .name = "fc1",
        .op_count = 2ULL * BATCH_SIZE * FC1_IN_FEATURES * FC1_OUT_FEATURES,
        .bytes_in = POOLED_BYTES,
        .bytes_weights = sizeof(fc1_weight),
        .bytes_out = FC1_OUT_BYTES,
    };
    profiles[PROFILE_FC2] = (layer_profile_t){
        .name = "fc2",
        .op_count = 2ULL * BATCH_SIZE * FC2_IN_FEATURES * FC2_OUT_FEATURES,
        .bytes_in = FC1_OUT_BYTES,
        .bytes_weights = sizeof(fc2_weight),
        .bytes_out = FC2_OUT_BYTES,
    };
}

static void print_setup_section(void) {
    printf("== Setup and Fairness ==\n");
    printf("model,sentence_gemmini_i8\n");
    printf("precision,int8\n");
    printf("batch_size,%u\n", BATCH_SIZE);
    printf("dataset,single_sample_only\n");
    printf("input_bytes,%u\n", (unsigned)sizeof(input));
    printf("backend,Gemmini\n");
    printf("runner,conv1d_direct_v1\n");
    printf("native_conv,%u\n", GEMMINI_NATIVE_CONV);
    printf("requantization,%s\n", GEMMINI_PER_TENSOR_REQUANT ? "gemmini_per_tensor_experimental" : "cpu_per_channel_symmetric");
    printf("pooling,%s\n", GEMMINI_STAGED_POOLING ? "gemmini_staged_identity_conv_pool_proxy" :
        (GEMMINI_TIMING_SKIP_CPU_POOL ? "timing_skip_cpu_pool_first_token_placeholder" : "cpu_global_maxpool"));
}

static void print_end_to_end_latency(uint64_t total_cycles, int prediction, int correct)
{
    printf("== End-to-End Latency ==\n");
    printf("run,single_sample,backend,gemmini,total_cycles,%lu,total_latency_s,%.6f,prediction,%d,correct,%d/1\n",
           (unsigned long)total_cycles,
           (double)total_cycles / (double)BAREMETAL_CPU_HZ,
           prediction,
           correct);
}

static void print_activation_summary(const layer_profile_t profiles[PROFILE_COUNT], uint64_t final_activation_cycles, uint64_t total_cycles)
{
    uint64_t total_compute_cycles = 0;
    uint64_t total_requantize_cycles = 0;
    uint64_t total_activation_fn_cycles = 0;
    uint64_t total_pool_cycles = 0;

    for (int i = 0; i < PROFILE_COUNT; ++i) {
        total_compute_cycles += profiles[i].compute_cycles;
        total_requantize_cycles += profiles[i].requantize_cycles;
        total_activation_fn_cycles += profiles[i].activation_cycles;
        total_pool_cycles += profiles[i].pool_cycles;
    }

#if GEMMINI_STAGED_POOLING
    const uint64_t gemmini_pool_cycles = total_pool_cycles;
    const uint64_t host_pool_cycles = 0;
#elif GEMMINI_TIMING_SKIP_CPU_POOL
    const uint64_t gemmini_pool_cycles = 0;
    const uint64_t host_pool_cycles = 0;
#else
    const uint64_t gemmini_pool_cycles = 0;
    const uint64_t host_pool_cycles = total_pool_cycles;
#endif

    const uint64_t gemmini_total_cycles = total_compute_cycles + gemmini_pool_cycles;
    const uint64_t host_postprocess_cycles =
        total_requantize_cycles + total_activation_fn_cycles + host_pool_cycles + final_activation_cycles;
    const uint64_t measured_accounted_cycles = gemmini_total_cycles + host_postprocess_cycles;
    const double compute_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)total_compute_cycles / (double)total_cycles;
    const double gemmini_pool_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)gemmini_pool_cycles / (double)total_cycles;
    const double gemmini_total_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)gemmini_total_cycles / (double)total_cycles;
    const double host_postprocess_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)host_postprocess_cycles / (double)total_cycles;
    const double requantize_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)total_requantize_cycles / (double)total_cycles;
    const double activation_fn_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)total_activation_fn_cycles / (double)total_cycles;
    const double host_pool_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)host_pool_cycles / (double)total_cycles;
    const double final_activation_percent = total_cycles == 0 ? 0.0 :
        100.0 * (double)final_activation_cycles / (double)total_cycles;

    printf("== Gemmini vs Host Breakdown ==\n");
    printf("summary,gemmini_compute_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)total_compute_cycles,
           (double)total_compute_cycles / (double)BAREMETAL_CPU_HZ,
           compute_percent);
    printf("summary,gemmini_pool_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)gemmini_pool_cycles,
           (double)gemmini_pool_cycles / (double)BAREMETAL_CPU_HZ,
           gemmini_pool_percent);
    printf("summary,gemmini_total_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)gemmini_total_cycles,
           (double)gemmini_total_cycles / (double)BAREMETAL_CPU_HZ,
           gemmini_total_percent);
    printf("summary,host_requantize_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)total_requantize_cycles,
           (double)total_requantize_cycles / (double)BAREMETAL_CPU_HZ,
           requantize_percent);
    printf("summary,host_activation_fn_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)total_activation_fn_cycles,
           (double)total_activation_fn_cycles / (double)BAREMETAL_CPU_HZ,
           activation_fn_percent);
    printf("summary,host_pool_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)host_pool_cycles,
           (double)host_pool_cycles / (double)BAREMETAL_CPU_HZ,
           host_pool_percent);
    printf("summary,host_final_sigmoid_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)final_activation_cycles,
           (double)final_activation_cycles / (double)BAREMETAL_CPU_HZ,
           final_activation_percent);
    printf("summary,host_postprocess_cycles,%lu,latency_s,%.6f,percent,%.3f\n",
           (unsigned long)host_postprocess_cycles,
           (double)host_postprocess_cycles / (double)BAREMETAL_CPU_HZ,
           host_postprocess_percent);
    printf("summary,accounted_cycles,%lu,latency_s,%.6f,note,gemmini_total_plus_host_postprocess\n",
           (unsigned long)measured_accounted_cycles,
           (double)measured_accounted_cycles / (double)BAREMETAL_CPU_HZ);
}

static void print_per_layer_breakdown(const layer_profile_t profiles[PROFILE_COUNT], uint64_t total_cycles)
{
    printf("== Per-Layer Breakdown ==\n");
    for (int i = 0; i < PROFILE_COUNT; ++i) {
        double percent = total_cycles == 0 ? 0.0 :
            100.0 * (double)profiles[i].cycles / (double)total_cycles;
        printf("layer,%d,backend,gemmini,name,%s,type,%s,cycles,%lu,latency_s,%.6f,percent,%.3f,compute_cycles,%lu,requantize_cycles,%lu,activation_cycles,%lu,pool_cycles,%lu\n",
               i,
               profiles[i].name,
               profiles[i].name,
               (unsigned long)profiles[i].cycles,
               (double)profiles[i].cycles / (double)BAREMETAL_CPU_HZ,
               percent,
               (unsigned long)profiles[i].compute_cycles,
               (unsigned long)profiles[i].requantize_cycles,
               (unsigned long)profiles[i].activation_cycles,
               (unsigned long)profiles[i].pool_cycles);
    }
}

static void print_roofline_section(const layer_profile_t profiles[PROFILE_COUNT])
{
    printf("== Roofline / Bottleneck Analysis ==\n");
    for (int i = 0; i < PROFILE_COUNT; ++i) {
        size_t total_bytes = profiles[i].bytes_in + profiles[i].bytes_weights + profiles[i].bytes_out;
        double intensity = total_bytes == 0 ? 0.0 :
            (double)profiles[i].op_count / (double)total_bytes;
        printf("layer,%d,backend,gemmini,ops,%lu,bytes,%lu,op_intensity,%.6f\n",
               i,
               (unsigned long)profiles[i].op_count,
               (unsigned long)total_bytes,
               intensity);
    }
}

static void print_memory_traffic_section(const layer_profile_t profiles[PROFILE_COUNT])
{
    printf("== Memory Traffic Analysis ==\n");
    for (int i = 0; i < PROFILE_COUNT; ++i) {
        size_t total_bytes = profiles[i].bytes_in + profiles[i].bytes_weights + profiles[i].bytes_out;
        printf("layer,%d,backend,gemmini,input_bytes,%lu,weight_bytes,%lu,output_bytes,%lu,total_bytes,%lu\n",
               i,
               (unsigned long)profiles[i].bytes_in,
               (unsigned long)profiles[i].bytes_weights,
               (unsigned long)profiles[i].bytes_out,
               (unsigned long)total_bytes);
    }
}

static void print_scaling_placeholder(void)
{
    printf("== Scaling Trends ==\n");
    printf("note,Single-sample Gemmini app currently has fixed model/input dimensions. Rebuild with shape variants for sweep data.\n");
}

static void print_operator_coverage(void)
{
    printf("== Operator Coverage and Fallback ==\n");
    printf("operator,CONV1D,gemmini,direct\n");
#if GEMMINI_NATIVE_CONV
    printf("operator,CONV1D_NATIVE,gemmini,tiled_conv_auto_hwio\n");
#endif
#if GEMMINI_PER_TENSOR_REQUANT
    printf("operator,REQUANTIZE,gemmini,per_tensor_store_scale\n");
    printf("operator,RELU,gemmini,store_activation\n");
#else
    printf("operator,REQUANTIZE,cpu,per_channel_symmetric\n");
    printf("operator,RELU,cpu,postprocess\n");
#endif
#if GEMMINI_STAGED_POOLING
    printf("operator,POOL1D,gemmini,staged_identity_conv_pool_proxy\n");
#elif GEMMINI_TIMING_SKIP_CPU_POOL
    printf("operator,POOL1D,gemmini,timing_placeholder_no_cpu_pool\n");
#else
    printf("operator,POOL1D,cpu,global_maxpool\n");
#endif
    printf("operator,FC,%s,%s\n", GEMMINI_FC_ON_CPU ? "cpu" : "gemmini",
        GEMMINI_FC_ON_CPU ? "int8_linear_reference" : "direct");
    printf("operator,SIGMOID,gemmini,host_postprocess\n");
}

static void im2col_1d(int batch_size, int in_col_dim, int in_channels,
    int kernel_size, int padding, int stride, int out_col_dim,
    const elem_t *input, elem_t *output)
{
    const int k_dim = kernel_size * in_channels;

    for (int b = 0; b < batch_size; b++) {
        for (int ocol = 0; ocol < out_col_dim; ocol++) {
            const int row = b * out_col_dim + ocol;

            for (int k = 0; k < kernel_size; k++) {
                const int in_col = ocol * stride + k - padding;
                for (int ch = 0; ch < in_channels; ch++) {
                    elem_t val = 0;
                    if (in_col >= 0 && in_col < in_col_dim) {
                        const size_t idx = (size_t)(b * in_col_dim + in_col) * in_channels + ch;
                        val = input[idx];
                    }
                    output[row * k_dim + (k * in_channels + ch)] = val;
                }
            }
        }
    }
}

static int8_t requantize_int8(int32_t input, float scale, int32_t zero_point);

static int conv1d_last_valid_kernel(int col, int in_col_dim, int kernel_size, int padding)
{
    int last = in_col_dim + padding - 1 - col;
    if (last >= kernel_size) {
        last = kernel_size - 1;
    }
    return last;
}

static void tiled_conv1d_direct_fused_auto(
    int batch_size, int in_col_dim, int in_channels,
    int out_channels, int kernel_size, int padding, int stride, int out_col_dim,
    const elem_t *input, const elem_t *weights_mat, const acc_t *bias,
    acc_t *output_acc, elem_t *output, acc_scale_t output_scale, int act)
{
    assert(stride == 1);

    for (int b = 0; b < batch_size; b++) {
        for (int ocol = 0; ocol < out_col_dim; ocol++) {
            acc_t *out_row = output_acc + ((size_t)b * out_col_dim + ocol) * out_channels;
            for (int och = 0; och < out_channels; och++) {
                out_row[och] = bias == NULL ? 0 : bias[och];
            }
        }
    }

    for (int k = 0; k < kernel_size; k++) {
        const int start_ocol = padding - k > 0 ? padding - k : 0;
        const int end_ocol_bound = in_col_dim + padding - k;
        const int end_ocol = end_ocol_bound < out_col_dim ? end_ocol_bound : out_col_dim;
        const int rows = end_ocol - start_ocol;

        if (rows <= 0) {
            continue;
        }

        const elem_t *weights_slice = weights_mat + (size_t)k * in_channels * out_channels;

        for (int b = 0; b < batch_size; b++) {
            const int in_col = start_ocol + k - padding;
            const elem_t *input_slice = input + ((size_t)b * in_col_dim + in_col) * in_channels;
            acc_t *output_slice = output_acc + ((size_t)b * out_col_dim + start_ocol) * out_channels;
            elem_t *output_elem_slice = output + ((size_t)b * out_col_dim + start_ocol) * out_channels;
            const int final_start_ocol = in_col_dim + padding - 1 - k;
            const int final_start = final_start_ocol > start_ocol ? final_start_ocol : start_ocol;
            const int nonfinal_rows = final_start < end_ocol ? final_start - start_ocol : rows;
            const int final_rows = rows - nonfinal_rows;

            if (output == NULL) {
                tiled_matmul_auto(rows, out_channels, in_channels,
                    input_slice, weights_slice,
                    output_slice, output_slice,
                    in_channels, out_channels, out_channels, out_channels,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                    false,
                    false, false,
                    true, false,
                    0,
                    WS);
                continue;
            }

            if (nonfinal_rows > 0) {
                tiled_matmul_auto(nonfinal_rows, out_channels, in_channels,
                    input_slice, weights_slice,
                    output_slice, output_slice,
                    in_channels, out_channels, out_channels, out_channels,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
                    false,
                    false, false,
                    true, false,
                    0,
                    WS);
            }

            if (final_rows > 0) {
                const elem_t *final_input_slice = input_slice + (size_t)nonfinal_rows * in_channels;
                acc_t *final_acc_slice = output_slice + (size_t)nonfinal_rows * out_channels;
                elem_t *final_output_slice = output_elem_slice + (size_t)nonfinal_rows * out_channels;

                assert(conv1d_last_valid_kernel(start_ocol + nonfinal_rows, in_col_dim, kernel_size, padding) == k);
                tiled_matmul_auto(final_rows, out_channels, in_channels,
                    final_input_slice, weights_slice,
                    final_acc_slice, final_output_slice,
                    in_channels, out_channels, out_channels, out_channels,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    act, output_scale, 0,
                    false,
                    false, false,
                    false, false,
                    0,
                    WS);
            }
        }
    }
}

static void flatten_conv_weights_1d(int out_channels, int in_channels, int kernel_size,
    const elem_t *weights, elem_t *weights_mat)
{
    const int k_dim = kernel_size * in_channels;

    for (int och = 0; och < out_channels; och++) {
        for (int k = 0; k < kernel_size; k++) {
            for (int ch = 0; ch < in_channels; ch++) {
                const int k_idx = k * in_channels + ch;
                const size_t w_idx = (((size_t)och * kernel_size + k) * in_channels + ch);
                weights_mat[k_idx * out_channels + och] = weights[w_idx];
            }
        }
    }
}

static void flatten_fc_weights(int out_features, int in_features,
    const elem_t *weights, elem_t *weights_mat)
{
    for (int out = 0; out < out_features; out++) {
        for (int in = 0; in < in_features; in++) {
            const size_t w_idx = (size_t)out * in_features + in;
            weights_mat[in * out_features + out] = weights[w_idx];
        }
    }
}

static int32_t round_to_int(float x) {
    return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

static int8_t clip_i32_i8(int32_t x) {
    if (x < -127) {
        return -127;
    }
    if (x > 127) {
        return 127;
    }
    return (int8_t)x;
}

static int8_t requantize_int8(int32_t input, float scale, int32_t zero_point) {
    int32_t q = round_to_int((float)input * scale) + zero_point;
    return clip_i32_i8(q);
}

static float sigmoidf(float x) {
    if (x <= -8.0f) {
        return 0.0f;
    }
    if (x >= 8.0f) {
        return 1.0f;
    }

    const float x2 = x * x;
    const float x3 = x2 * x;
    const float x5 = x3 * x2;
    float y = 0.5f + (x * 0.25f) - (x3 * (1.0f / 48.0f)) + (x5 * (1.0f / 480.0f));
    return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
}

static void gemmini_conv1d(
    const elem_t *input_nhwc,
    const elem_t *weights_hwio,
    const elem_t *weights_mat,
    const acc_t *bias,
    const float *per_channel_m,
    const elem_t *zero_points,
    elem_t *output_nhwc,
    acc_t *acc_buf,
    int batch_size,
    int in_row_dim,
    int in_col_dim,
    int in_channels,
    int out_channels,
    int out_col_dim,
    int kernel_size,
    int padding,
    int stride,
    acc_scale_t per_tensor_scale,
    layer_profile_t *profile)
{
    uint64_t layer_start = read_rdcycle();
    uint64_t stage_start = layer_start;

#if GEMMINI_NATIVE_CONV
    tiled_conv_auto(batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, in_row_dim, out_col_dim,
        stride, 1, 1, padding, kernel_size,
        false, false, false,
        false, false,
        input_nhwc, weights_hwio, bias,
        output_nhwc,
        RELU, per_tensor_scale,
        1, 0, 0,
        WS);
    profile->compute_cycles = read_rdcycle() - stage_start;
#else
#if GEMMINI_PER_TENSOR_REQUANT
    tiled_conv1d_direct_fused_auto(batch_size, in_col_dim, in_channels,
        out_channels, kernel_size, padding, stride, out_col_dim,
        input_nhwc, weights_mat, bias,
        acc_buf, output_nhwc, per_tensor_scale, RELU);
    profile->compute_cycles = read_rdcycle() - stage_start;
#else
    tiled_conv1d_direct_fused_auto(batch_size, in_col_dim, in_channels,
        out_channels, kernel_size, padding, stride, out_col_dim,
        input_nhwc, weights_mat, bias,
        acc_buf, NULL, ACC_SCALE_IDENTITY, NO_ACTIVATION);
    profile->compute_cycles = read_rdcycle() - stage_start;

    stage_start = read_rdcycle();
    for (int b = 0; b < batch_size; b++) {
        for (int col = 0; col < out_col_dim; col++) {
            const int row = b * out_col_dim + col;
            for (int ch = 0; ch < out_channels; ch++) {
                output_nhwc[((size_t)b * out_col_dim + col) * out_channels + ch] =
                    requantize_int8(acc_buf[(size_t)row * out_channels + ch], per_channel_m[ch], zero_points[ch]);
            }
        }
    }
    profile->requantize_cycles = read_rdcycle() - stage_start;

    stage_start = read_rdcycle();
    for (int b = 0; b < batch_size; b++) {
        for (int col = 0; col < out_col_dim; col++) {
            for (int ch = 0; ch < out_channels; ch++) {
                elem_t *q = &output_nhwc[((size_t)b * out_col_dim + col) * out_channels + ch];
                if (*q < 0) {
                    *q = 0;
                }
            }
        }
    }
    profile->activation_cycles = read_rdcycle() - stage_start;
#endif
#endif
    profile->cycles = read_rdcycle() - layer_start;
}

static const elem_t *gemmini_pool1d_global_or_staged(
    const elem_t *conv3_out_nhwc,
    elem_t *pooled,
    elem_t *pool_stage1,
    elem_t *pool_stage2,
    elem_t *pool_stage3,
    layer_profile_t *profile)
{
    uint64_t layer_start = read_rdcycle();
    uint64_t stage_start = layer_start;
    const elem_t *fc1_input = pooled;

#if GEMMINI_STAGED_POOLING
    tiled_conv_auto(BATCH_SIZE, 16, 24, CONV3_OUT_CHANNELS,
        CONV3_OUT_CHANNELS, 16, 24,
        1, 1, 1, 0, 1,
        false, false, false,
        false, false,
        conv3_out_nhwc, (const elem_t *)pool_identity_1x1_hwio, NULL,
        pool_stage1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY,
        2, 2, 0,
        WS);
    tiled_conv_auto(BATCH_SIZE, 8, 12, CONV3_OUT_CHANNELS,
        CONV3_OUT_CHANNELS, 8, 12,
        1, 1, 1, 0, 1,
        false, false, false,
        false, false,
        pool_stage1, (const elem_t *)pool_identity_1x1_hwio, NULL,
        pool_stage2,
        NO_ACTIVATION, ACC_SCALE_IDENTITY,
        2, 2, 0,
        WS);
    tiled_conv_auto(BATCH_SIZE, 4, 6, CONV3_OUT_CHANNELS,
        CONV3_OUT_CHANNELS, 4, 6,
        1, 1, 1, 0, 1,
        false, false, false,
        false, false,
        pool_stage2, (const elem_t *)pool_identity_1x1_hwio, NULL,
        pool_stage3,
        NO_ACTIVATION, ACC_SCALE_IDENTITY,
        2, 2, 0,
        WS);
    fc1_input = pool_stage3;
    profile->pool_cycles = read_rdcycle() - stage_start;
#elif GEMMINI_TIMING_SKIP_CPU_POOL
    fc1_input = conv3_out_nhwc;
    profile->pool_cycles = 0;
#else
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int ch = 0; ch < CONV3_OUT_CHANNELS; ch++) {
            elem_t max_val = conv3_out_nhwc[((size_t)b * CONV3_OUT_COL_DIM) * CONV3_OUT_CHANNELS + ch];
            for (int col = 1; col < CONV3_OUT_COL_DIM; col++) {
                const elem_t q = conv3_out_nhwc[((size_t)b * CONV3_OUT_COL_DIM + col) * CONV3_OUT_CHANNELS + ch];
                if (q > max_val) {
                    max_val = q;
                }
            }
            pooled[(size_t)b * FC1_IN_FEATURES + ch] = max_val;
        }
    }
    profile->pool_cycles = read_rdcycle() - stage_start;
#endif
    profile->cycles = read_rdcycle() - layer_start;
    return fc1_input;
}

static void gemmini_linear(
    const elem_t *input,
    const elem_t *weights_mat,
    const acc_t *bias,
    const float *per_channel_m,
    const elem_t *zero_points,
    elem_t *output,
    acc_t *acc_buf,
    int in_features,
    int out_features,
    int act,
    acc_scale_t per_tensor_scale,
    layer_profile_t *profile)
{
    uint64_t layer_start = read_rdcycle();
    uint64_t stage_start = layer_start;

#if GEMMINI_FC_ON_CPU
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int out = 0; out < out_features; out++) {
            acc_t acc = bias == NULL ? 0 : bias[out];
            for (int in = 0; in < in_features; in++) {
                acc += (acc_t)input[(size_t)b * in_features + in] *
                       (acc_t)weights_mat[(size_t)in * out_features + out];
            }
            output[(size_t)b * out_features + out] =
                requantize_int8(acc, per_tensor_scale, 0);
        }
    }
    profile->compute_cycles = read_rdcycle() - stage_start;

    if (act == RELU) {
        stage_start = read_rdcycle();
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int out = 0; out < out_features; out++) {
                elem_t *q = &output[(size_t)b * out_features + out];
                if (*q < 0) {
                    *q = 0;
                }
            }
        }
        profile->activation_cycles = read_rdcycle() - stage_start;
    }
#else
#if GEMMINI_PER_TENSOR_REQUANT
    tiled_matmul_auto(BATCH_SIZE, out_features, in_features,
        input, weights_mat,
        bias, output,
        in_features, out_features,
        out_features, out_features,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, per_tensor_scale, 0,
        true,
        false, false,
        false, false,
        0,
        WS);
    profile->compute_cycles = read_rdcycle() - stage_start;
#else
    tiled_matmul_auto(BATCH_SIZE, out_features, in_features,
        input, weights_mat,
        bias, acc_buf,
        in_features, out_features,
        out_features, out_features,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    profile->compute_cycles = read_rdcycle() - stage_start;

    stage_start = read_rdcycle();
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int ch = 0; ch < out_features; ch++) {
            output[(size_t)b * out_features + ch] =
                requantize_int8(acc_buf[(size_t)b * out_features + ch], per_channel_m[ch], zero_points[ch]);
        }
    }
    profile->requantize_cycles = read_rdcycle() - stage_start;

    if (act == RELU) {
        stage_start = read_rdcycle();
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int ch = 0; ch < out_features; ch++) {
                elem_t *q = &output[(size_t)b * out_features + ch];
                if (*q < 0) {
                    *q = 0;
                }
            }
        }
        profile->activation_cycles = read_rdcycle() - stage_start;
    }
#endif
#endif
    profile->cycles = read_rdcycle() - layer_start;
}

int main() {
#if !defined(BAREMETAL) && !defined(SPIKE_PK)
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif

    gemmini_flush(0);

    layer_profile_t profiles[PROFILE_COUNT];
    init_profiles(profiles);

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Runner: conv1d_direct_v1\n");
    printf("Conv1 out cols: %u\n", CONV1_OUT_COL_DIM);
    printf("Conv2 out cols: %u\n", CONV2_OUT_COL_DIM);
    printf("Conv3 out cols: %u\n\n", CONV3_OUT_COL_DIM);
    print_setup_section();

    static elem_t conv1_weights_mat[CONV1_KERNEL * CONV1_IN_CHANNELS][CONV1_OUT_CHANNELS];
    static elem_t conv2_weights_mat[CONV2_KERNEL * CONV2_IN_CHANNELS][CONV2_OUT_CHANNELS];
    static elem_t conv3_weights_mat[CONV3_KERNEL * CONV3_IN_CHANNELS][CONV3_OUT_CHANNELS];
    static elem_t fc1_weights_mat[FC1_IN_FEATURES][FC1_OUT_FEATURES];
    static elem_t fc2_weights_mat[FC2_IN_FEATURES][FC2_OUT_FEATURES];

    static acc_t conv1_acc[BATCH_SIZE * CONV1_OUT_COL_DIM][CONV1_OUT_CHANNELS];
    static acc_t conv2_acc[BATCH_SIZE * CONV2_OUT_COL_DIM][CONV2_OUT_CHANNELS];
    static acc_t conv3_acc[BATCH_SIZE * CONV3_OUT_COL_DIM][CONV3_OUT_CHANNELS];
    static acc_t fc1_acc[BATCH_SIZE][FC1_OUT_FEATURES];
    static acc_t fc2_acc[BATCH_SIZE][FC2_OUT_FEATURES];

    static elem_t conv1_out[BATCH_SIZE][1][CONV1_OUT_COL_DIM][CONV1_OUT_CHANNELS];
    static elem_t conv2_out[BATCH_SIZE][1][CONV2_OUT_COL_DIM][CONV2_OUT_CHANNELS];
    static elem_t conv3_out[BATCH_SIZE][1][CONV3_OUT_COL_DIM][CONV3_OUT_CHANNELS];

    static elem_t pooled[BATCH_SIZE][FC1_IN_FEATURES];
    static elem_t pool_stage1[BATCH_SIZE][8][12][CONV3_OUT_CHANNELS];
    static elem_t pool_stage2[BATCH_SIZE][4][6][CONV3_OUT_CHANNELS];
    static elem_t pool_stage3[BATCH_SIZE][2][3][CONV3_OUT_CHANNELS];
    static elem_t fc1_out[BATCH_SIZE][FC1_OUT_FEATURES];
    static elem_t fc2_out[BATCH_SIZE][FC2_OUT_FEATURES];

    flatten_conv_weights_1d(CONV1_OUT_CHANNELS, CONV1_IN_CHANNELS, CONV1_KERNEL,
        (const elem_t *)conv1_weight, (elem_t *)conv1_weights_mat);
    flatten_conv_weights_1d(CONV2_OUT_CHANNELS, CONV2_IN_CHANNELS, CONV2_KERNEL,
        (const elem_t *)conv2_weight, (elem_t *)conv2_weights_mat);
    flatten_conv_weights_1d(CONV3_OUT_CHANNELS, CONV3_IN_CHANNELS, CONV3_KERNEL,
        (const elem_t *)conv3_weight, (elem_t *)conv3_weights_mat);
    flatten_fc_weights(FC1_OUT_FEATURES, FC1_IN_FEATURES,
        (const elem_t *)fc1_weight, (elem_t *)fc1_weights_mat);
    flatten_fc_weights(FC2_OUT_FEATURES, FC2_IN_FEATURES,
        (const elem_t *)fc2_weight, (elem_t *)fc2_weights_mat);

#if GEMMINI_ACCURACY_SWEEP
    uint64_t sweep_cycles_start = read_rdcycle();
    int correct_count = 0;
    int positive_predictions = 0;
    int positive_labels = 0;

    for (int sample_idx = 0; sample_idx < SENTENCE_DATASET_SIZE; sample_idx++) {
        init_profiles(profiles);
        const elem_t *current_input = (const elem_t *)sentence_inputs[sample_idx];
        const bool current_valid = sentence_labels[sample_idx];
        positive_labels += current_valid ? 1 : 0;
#else
    const elem_t *current_input = (const elem_t *)input;
    const bool current_valid = valid_embedding;
#endif

    uint64_t total_cycles_start = read_rdcycle();
    gemmini_conv1d(current_input,
        (const elem_t *)conv1_weight_hwio, (const elem_t *)conv1_weights_mat,
        (const acc_t *)conv1_bias, conv1_M, conv1_zero_points,
        (elem_t *)conv1_out, (acc_t *)conv1_acc,
        BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS,
        CONV1_OUT_COL_DIM, CONV1_KERNEL, CONV1_PADDING, CONV1_STRIDE,
        CONV1_REQUANT_SCALE, &profiles[PROFILE_CONV1]);

    gemmini_conv1d((const elem_t *)conv1_out,
        (const elem_t *)conv2_weight_hwio, (const elem_t *)conv2_weights_mat,
        (const acc_t *)conv2_bias, conv2_M, conv2_zero_points,
        (elem_t *)conv2_out, (acc_t *)conv2_acc,
        BATCH_SIZE, IN_ROW_DIM, CONV1_OUT_COL_DIM, CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS,
        CONV2_OUT_COL_DIM, CONV2_KERNEL, CONV2_PADDING, CONV2_STRIDE,
        CONV2_REQUANT_SCALE, &profiles[PROFILE_CONV2]);

    gemmini_conv1d((const elem_t *)conv2_out,
        (const elem_t *)conv3_weight_hwio, (const elem_t *)conv3_weights_mat,
        (const acc_t *)conv3_bias, conv3_M, conv3_zero_points,
        (elem_t *)conv3_out, (acc_t *)conv3_acc,
        BATCH_SIZE, IN_ROW_DIM, CONV2_OUT_COL_DIM, CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS,
        CONV3_OUT_COL_DIM, CONV3_KERNEL, CONV3_PADDING, CONV3_STRIDE,
        CONV3_REQUANT_SCALE, &profiles[PROFILE_CONV3]);

    const elem_t *fc1_input = gemmini_pool1d_global_or_staged((const elem_t *)conv3_out,
        (elem_t *)pooled, (elem_t *)pool_stage1, (elem_t *)pool_stage2, (elem_t *)pool_stage3,
        &profiles[PROFILE_POOL]);

    gemmini_linear(fc1_input, (elem_t *)fc1_weights_mat,
        (const acc_t *)fc1_bias, fc1_M, fc1_zero_points,
        (elem_t *)fc1_out, (acc_t *)fc1_acc,
        FC1_IN_FEATURES, FC1_OUT_FEATURES, RELU, FC1_REQUANT_SCALE,
        &profiles[PROFILE_FC1]);

    gemmini_linear((elem_t *)fc1_out, (elem_t *)fc2_weights_mat,
        (const acc_t *)fc2_bias, fc2_M, fc2_zero_points,
        (elem_t *)fc2_out, (acc_t *)fc2_acc,
        FC2_IN_FEATURES, FC2_OUT_FEATURES, NO_ACTIVATION, FC2_REQUANT_SCALE,
        &profiles[PROFILE_FC2]);

    uint64_t final_activation_start = read_rdcycle();
    float logit = (float)fc2_out[0][0] * FC2_OUT_SCALE;
    float prob = sigmoidf(logit);
    uint64_t final_activation_cycles = read_rdcycle() - final_activation_start;
    uint64_t total_cycles_end = read_rdcycle();
    uint64_t total_cycles = total_cycles_end - total_cycles_start;
    bool correct = (prob >= 0.5f) ? true : false; // valid embedding/invalid embedding

#if GEMMINI_ACCURACY_SWEEP
        positive_predictions += correct ? 1 : 0;
        if (correct == current_valid) {
            correct_count++;
        }
    }

    uint64_t sweep_cycles = read_rdcycle() - sweep_cycles_start;
    float accuracy = SENTENCE_DATASET_SIZE == 0 ? 0.0f : (float)correct_count / (float)SENTENCE_DATASET_SIZE;
    printf("== Accuracy Sweep ==\n");
    printf("dataset,sentence_dataset_gemmini\n");
    printf("samples,%d\n", SENTENCE_DATASET_SIZE);
    printf("correct,%d\n", correct_count);
    printf("accuracy,%.6f\n", accuracy);
    printf("positive_labels,%d\n", positive_labels);
    printf("positive_predictions,%d\n", positive_predictions);
    printf("total_cycles,%lu\n", sweep_cycles);
    printf("avg_cycles_per_sample,%lu\n", SENTENCE_DATASET_SIZE == 0 ? 0UL : sweep_cycles / (uint64_t)SENTENCE_DATASET_SIZE);
    printf("avg_latency_s,%.6f\n", SENTENCE_DATASET_SIZE == 0 ? 0.0 :
        (double)(sweep_cycles / (uint64_t)SENTENCE_DATASET_SIZE) / (double)BAREMETAL_CPU_HZ);
    return 0;
#else
    print_end_to_end_latency(total_cycles, fc2_out[0][0], current_valid == correct);
    print_activation_summary(profiles, final_activation_cycles, total_cycles);
    print_per_layer_breakdown(profiles, total_cycles);
    print_roofline_section(profiles);
    print_memory_traffic_section(profiles);
    print_scaling_placeholder();
    print_operator_coverage();

    printf("Output logit (int8): %d\n", fc2_out[0][0]);
    printf("Output sigmoid: %f\n", prob);
    if(correct == current_valid) {
        printf("Correctly classified\n");
    } else {
        printf("Incorrectly classified\n");
    }

    return 0;
#endif
}
