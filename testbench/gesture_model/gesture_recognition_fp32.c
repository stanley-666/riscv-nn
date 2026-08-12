/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_layer.h"
#include "nn_runtime.h"
#include "nn_utils.h"
#if NN_BACKEND_CPU
#include "nn_infer_cpu.h"
#endif
#if NN_BACKEND_VECTOR
#include "nn_infer_vpu.h"
#endif
#include "nn_ops.h"

#if NN_BACKEND_VECTOR
#define RUN_FORWARD(model, input) forward_fp32_vpu((model), (input))
#define RUN_SOFTMAX(input, output, size) softmax_f32_rvv((input), (output), (size))
#else
#define RUN_FORWARD(model, input) forward((model), (input))
#define RUN_SOFTMAX(input, output, size) softmax_f32((input), (output), (size))
#endif

#include "gesture_input.h"
#include "weights_fused_fp32.h"

#define GESTURE_MAX_ELEMS 12000  // max tensor elements across the model (float)
#define SOFTMAX_WARMUP_RUNS 5
#define SOFTMAX_MEASURED_RUNS 100
static float buffer1_static[GESTURE_MAX_ELEMS] __attribute__((aligned(64)));
static float buffer2_static[GESTURE_MAX_ELEMS] __attribute__((aligned(64)));

void init_pingpong_buffer(size_t num_elem, elem_type dtype) {
    (void)num_elem;
    (void)dtype;
    size_t size = GESTURE_MAX_ELEMS * sizeof(float);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

#ifdef WEIGHTS_FUSED_FP32_H
    void gesture_all_1dcnn_f32() {
        printf("1D CNN Inference Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(GESTURE_MAX_ELEMS, ELEM_FLOAT32);
        printf("Defining layers...\n");
        uint64_t layer_def_start = nn_runtime_read_cycles();
        NNModule *transpose_input = nn_Transpose(50, 5, TRANSPOSE_CW_TO_WC, ELEM_FLOAT32);
        NNModule *conv1 = nn_Conv1d(5, 50, 3, 32, 1, 0, RELU, conv1d_1_weight, conv1d_1_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv2 = nn_Conv1d(32, conv1->outputShape.W, 3, 64, 1, 0, RELU, conv1d_2_weight, conv1d_2_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv3 = nn_Conv1d(64, conv2->outputShape.W, 3, 128, 1, 0, RELU, conv1d_3_weight, conv1d_3_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv4 = nn_Conv1d(128, conv3->outputShape.W, 3, 256, 1, 0, RELU, conv1d_4_weight, conv1d_4_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv5 = nn_Conv1d(256, conv4->outputShape.W, 1, 256, 1, 0, RELU, conv1d_5_weight, conv1d_5_bias, NULL, NULL, ELEM_FLOAT32);
        int flatten_len = conv5->outputShape.C * conv5->outputShape.W;
        NNModule *fc = nn_Linear(flatten_len, 4, SOFTMAX, softmax_1_weight, softmax_1_bias, NULL, NULL, ELEM_FLOAT32);
        uint64_t layer_def_end = nn_runtime_read_cycles();
        double layer_def_elapsed = nn_runtime_cycles_to_seconds(layer_def_end - layer_def_start);
        printf("Layer definition time: %.6f seconds\n", layer_def_elapsed);
        CNN* model = createCNN();
        uint64_t add_layer_start = nn_runtime_read_cycles();

        addLayer(model, transpose_input);
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, conv4);
        addLayer(model, conv5);
        addLayer(model, fc);
        uint64_t add_layer_end = nn_runtime_read_cycles();
        double add_layer_elapsed = nn_runtime_cycles_to_seconds(add_layer_end - add_layer_start);
        printf("Layer addition time: %.6f seconds\n", add_layer_elapsed);
        forward_input_bytes = 250 * sizeof(float);

        const float *gestures[] = { gesture_1, gesture_2, gesture_3 };
        const char *gesture_names[] = { "gesture_1", "gesture_2", "gesture_3" };
        for (int g = 0; g < 3; ++g) {
            uint64_t start_cycle = nn_runtime_read_cycles();
            RUN_FORWARD(model, (void *)gestures[g]);
            RUN_SOFTMAX((float*)buffer2, (float*)buffer1, 4);
            uint64_t end_cycle = nn_runtime_read_cycles();
            float* output = (float *) buffer1;
            float reference[4];
            float max_error = 0.0f;
            float probability_sum = 0.0f;
            softmax_f32((float *)buffer2, reference, 4);
            for (int i = 0; i < 4; ++i) {
                float error = output[i] - reference[i];
                if (error < 0.0f) error = -error;
                if (error > max_error) max_error = error;
                probability_sum += output[i];
            }
            for (int i = 0; i < SOFTMAX_WARMUP_RUNS; ++i) {
                RUN_SOFTMAX((float *)buffer2, output, 4);
                softmax_f32((float *)buffer2, reference, 4);
            }
            uint64_t rvv_softmax_start = nn_runtime_read_cycles();
            for (int i = 0; i < SOFTMAX_MEASURED_RUNS; ++i)
                RUN_SOFTMAX((float *)buffer2, output, 4);
            uint64_t rvv_softmax_cycles = nn_runtime_read_cycles() - rvv_softmax_start;
            uint64_t expf_softmax_start = nn_runtime_read_cycles();
            for (int i = 0; i < SOFTMAX_MEASURED_RUNS; ++i)
                softmax_f32((float *)buffer2, reference, 4);
            uint64_t expf_softmax_cycles = nn_runtime_read_cycles() - expf_softmax_start;
            uint64_t cycle_diff = end_cycle - start_cycle;
            printf("[%s] Inference cycles: %lu cycles\n", gesture_names[g], cycle_diff);
            double elapsed_time = nn_runtime_cycles_to_seconds(cycle_diff);
            printf("[%s] Inference time: %.6f seconds\n", gesture_names[g], elapsed_time);
            printf("[%s] RVV Softmax sum: %.8f, max |RVV-reference|: %.8f\n",
                   gesture_names[g], probability_sum, max_error);
            printf("[%s] Softmax average: RVV=%lu cycles, expf=%lu cycles, difference=%ld cycles\n",
                   gesture_names[g],
                   (unsigned long)(rvv_softmax_cycles / SOFTMAX_MEASURED_RUNS),
                   (unsigned long)(expf_softmax_cycles / SOFTMAX_MEASURED_RUNS),
                   (long)(rvv_softmax_cycles / SOFTMAX_MEASURED_RUNS) -
                   (long)(expf_softmax_cycles / SOFTMAX_MEASURED_RUNS));
            //printf("[%s] Output vector (4): \ngesture_0_score : %.6f\ngesture_1_score : %.6f\ngesture_2_score : %.6f\ngesture_3_score : %.6f\n", gesture_names[g], output[0], output[1], output[2], output[3]);
            float sorted_scores[4];
            int sorted_idx[4];
            for (int i = 0; i < 4; ++i) {
                sorted_scores[i] = output[i];
                sorted_idx[i] = i;
            }
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 4; ++j) {
                    if (sorted_scores[j] > sorted_scores[i]) {
                        float tmp = sorted_scores[i];
                        sorted_scores[i] = sorted_scores[j];
                        sorted_scores[j] = tmp;
                        int tmp_idx = sorted_idx[i];
                        sorted_idx[i] = sorted_idx[j];
                        sorted_idx[j] = tmp_idx;
                    }
                }
            }
            printf("[%s] Sorted scores (high -> low):\n", gesture_names[g]);
            for (int i = 0; i < 4; ++i) {
                printf("  gesture_%d : %.6f\n", sorted_idx[i], sorted_scores[i]);
            }
            printf("[%s] Predicted gesture: gesture_%d with score %.6f\n", gesture_names[g], sorted_idx[0], sorted_scores[0]);
            
        }

        freeCNN(model);
    }
#endif

int main() {
    gesture_all_1dcnn_f32();
    return 0;
}
