/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */
#include "cnn_demo.h"
#include <stdio.h>
#include <string.h>
#include "baremetal_timer.h"
#include "gesture_input.h"
#include "model_builder.h"
#include "nn_infer_vpu.h"
#include "nn_ops.h"
#include "nn_utils.h"

#define GESTURE_MAX_ELEMS 12000
#define WARMUP_RUNS 5
#define MEASURED_RUNS 100
#define SOFTMAX_WARMUP_RUNS 5
#define SOFTMAX_MEASURED_RUNS 100
static float act_a[GESTURE_MAX_ELEMS] __attribute__((aligned(64)));
static float act_b[GESTURE_MAX_ELEMS] __attribute__((aligned(64)));

void run_rvv_single_demo(void)
{
    buffer1 = act_a;
    buffer2 = act_b;
    memset(act_a, 0, sizeof(act_a));
    memset(act_b, 0, sizeof(act_b));
    ModelBuildStats stats = {0};
    CNN *model = build_gesture_cnn(&stats);
    forward_input_bytes = 250 * sizeof(float);
    for (int i = 0; i < WARMUP_RUNS; ++i)
        forward_fp32_vpu(model, (void *)gesture_1);
    timer_ticks_t start = timer_now();
    for (int i = 0; i < MEASURED_RUNS; ++i)
        forward_fp32_vpu(model, (void *)gesture_1);
    timer_ticks_t total = timer_now() - start;
    float *raw = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
    float scores[4];
    softmax_f32_rvv(raw, scores, 4);
    float reference[4];
    float max_error = 0.0f;
    float probability_sum = 0.0f;
    softmax_f32(raw, reference, 4);
    for (int i = 0; i < 4; ++i) {
        float error = scores[i] - reference[i];
        if (error < 0.0f) error = -error;
        if (error > max_error) max_error = error;
        probability_sum += scores[i];
    }
    for (int i = 0; i < SOFTMAX_WARMUP_RUNS; ++i) {
        softmax_f32_rvv(raw, scores, 4);
        softmax_f32(raw, reference, 4);
    }
    timer_ticks_t rvv_softmax_start = timer_now();
    for (int i = 0; i < SOFTMAX_MEASURED_RUNS; ++i)
        softmax_f32_rvv(raw, scores, 4);
    timer_ticks_t rvv_softmax_cycles = timer_now() - rvv_softmax_start;
    timer_ticks_t expf_softmax_start = timer_now();
    for (int i = 0; i < SOFTMAX_MEASURED_RUNS; ++i)
        softmax_f32(raw, reference, 4);
    timer_ticks_t expf_softmax_cycles = timer_now() - expf_softmax_start;
    int prediction = 0;
    for (int i = 1; i < 4; ++i)
        if (scores[i] > scores[prediction]) prediction = i;
    printf("Average inference: %lu cycles (%d warm-up, %d measured)\n",
           (unsigned long)(total / MEASURED_RUNS), WARMUP_RUNS, MEASURED_RUNS);
    printf("Scores: %.6f %.6f %.6f %.6f\n", scores[0], scores[1], scores[2], scores[3]);
    printf("RVV Softmax sum: %.8f, max |RVV-reference|: %.8f\n",
           probability_sum, max_error);
    printf("Softmax average: RVV=%lu cycles, expf=%lu cycles, difference=%ld cycles\n",
           (unsigned long)(rvv_softmax_cycles / SOFTMAX_MEASURED_RUNS),
           (unsigned long)(expf_softmax_cycles / SOFTMAX_MEASURED_RUNS),
           (long)(rvv_softmax_cycles / SOFTMAX_MEASURED_RUNS) -
           (long)(expf_softmax_cycles / SOFTMAX_MEASURED_RUNS));
    printf("Predicted gesture: %d\n", prediction);
    freeCNN(model);
    buffer1 = NULL;
    buffer2 = NULL;
}
