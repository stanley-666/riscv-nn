/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cnn_demo.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "baremetal_timer.h"
#include "model_builder.h"
#include "nn_infer_vpu.h"
#include "nn_utils.h"
#include "random_embedding.h"

#define SENTENCE_INT8_MAX_ELEMS (384U * 256U)
#define BENCHMARK_WARMUP_RUNS 5
#define BENCHMARK_RUNS 100

static int8_t input_buffer[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));
static int8_t output_buffer[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));

static void init_static_pingpong_buffers(void)
{
    buffer1 = input_buffer;
    buffer2 = output_buffer;
    memset(buffer1, 0, sizeof(input_buffer));
    memset(buffer2, 0, sizeof(output_buffer));
}

void run_rvv_single_demo(void)
{
    ModelBuildStats stats = {0};
    CNN *model = build_sentence_int8_cnn(&stats);
    init_static_pingpong_buffers();
    forward_input_bytes = 384 * sizeof(int8_t);

    printf("NNModule definition time: %.6f seconds\n", stats.definition_seconds);
    printf("NNModule addition time : %.6f seconds\n", stats.addition_seconds);

    for (int i = 0; i < BENCHMARK_WARMUP_RUNS; ++i)
        forward_int8_vpu(model, (void *)random_embedding);

    timer_ticks_t start = timer_now();
    for (int i = 0; i < BENCHMARK_RUNS; ++i)
        forward_int8_vpu(model, (void *)random_embedding);
    timer_ticks_t total = timer_now() - start;

    const int8_t *output = (const int8_t *)
        ((model->numModules % 2 == 0) ? buffer1 : buffer2);
    printf("Average inference: %lu cycles (%d warm-up, %d measured)\n",
           (unsigned long)(total / BENCHMARK_RUNS),
           BENCHMARK_WARMUP_RUNS, BENCHMARK_RUNS);
    printf("Ground truth: %d\n", valid_embedding);
    printf("Prediction  : %d\n", output[0]);

    freeCNN(model);
    buffer1 = NULL;
    buffer2 = NULL;
}
