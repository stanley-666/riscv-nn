#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "nn_layer.h"
#include "nn_utils.h"
#include "nn_infer_cpu.h"
#include "nn_infer_vpu.h"

#include "weights_fp32.h"
#include "random_embedding.h"
#include "test_dataset_2658.h"
//#include "test_dataset.h"

#define SENTENCE_FP32_MAX_ELEMS (384 * 256)

static float buffer1_static[SENTENCE_FP32_MAX_ELEMS] __attribute__((aligned(64)));
static float buffer2_static[SENTENCE_FP32_MAX_ELEMS] __attribute__((aligned(64)));

static double now_seconds(void)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

void init_pingpong_buffer(size_t num_elem, elem_type dtype)
{
    (void)num_elem;
    (void)dtype;
    size_t size = SENTENCE_FP32_MAX_ELEMS * sizeof(float);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

#ifdef WEIGHTS_FP32_H
static void test_sentence_logit_fp32_time(void)
{
    printf("1D CNN Inference Demo\n");
    printf("Initializing ping-pong buffers...\n");
    init_pingpong_buffer(SENTENCE_FP32_MAX_ELEMS, ELEM_FLOAT32);
    printf("Defining layers...\n");

    double layer_def_start = now_seconds();
    NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_FLOAT32);
    NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, NULL, NULL, ELEM_FLOAT32);
    double layer_def_end = now_seconds();
    printf("Layer definition time: %.6f seconds\n", layer_def_end - layer_def_start);

    printf("Creating CNN model...\n");
    CNN *model = createCNN();

    printf("Adding layers to the model...\n");
    double add_layer_start = now_seconds();
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, maxpool);
    addLayer(model, fc1);
    addLayer(model, fc2);
    double add_layer_end = now_seconds();
    printf("Layer addition time: %.6f seconds\n", add_layer_end - add_layer_start);

    float embedding_f32[384];
    for (int i = 0; i < 384; ++i) {
        embedding_f32[i] = (float)random_embedding[i];
    }
    forward_input_bytes = 384 * sizeof(float);

    double start_time = now_seconds();
    forward_fp32_vpu(model, (void *)embedding_f32);
    double end_time = now_seconds();

    float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
    printf("Inference time: %.6f seconds\n", end_time - start_time);

    printf("Ground truth (valid_embedding): %d\n", valid_embedding);
    float prob = output[0];
    int pred = prob >= 0.5f;
    printf("Model prediction (prob): %.4f -> label %d\n", prob, pred);

    if (pred == valid_embedding) {
        printf("Prediction correct!\n");
    } else {
        printf("Prediction incorrect!\n");
    }

    if (pred == 1) {
        printf("-> Model predicts: Valid sentence\n");
    } else {
        printf("-> Model predicts: Invalid sentence\n");
    }

    freeCNN(model);
}

static void sentence_all_f32_time(void)
{
    printf("1D CNN Inference Demo\n");
    printf("Initializing ping-pong buffers...\n");
    init_pingpong_buffer(SENTENCE_FP32_MAX_ELEMS, ELEM_FLOAT32);
    printf("Defining layers...\n");

    NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_FLOAT32);
    NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, NULL, NULL, ELEM_FLOAT32);

    CNN *model = createCNN();
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, maxpool);
    addLayer(model, fc1);
    addLayer(model, fc2);
    printf("Completed model creation.\n");

    int correct = 0;
    double elapsed_time = 0.0;
    float embedding_f32[384];
    forward_input_bytes = 384 * sizeof(float);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        const int8_t *embedding = test_embeddings[i];
        bool label = test_labels[i];

        for (int j = 0; j < 384; ++j) {
            embedding_f32[j] = (float)embedding[j];
        }

        double start_time = now_seconds();
        forward_fp32_vpu(model, (void *)embedding_f32);
        double end_time = now_seconds();
        elapsed_time += end_time - start_time;

        float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
        float prob = output[0];
        int pred = (prob >= 0.5f) ? 1 : 0;

        if (pred == label) {
            correct++;
        } else {
            printf("[%d] ", i);
            for (int j = 0; j < 384; j++) {
                printf("%d ", embedding[j]);
            }
            printf("\n");
            printf("prob=%.4f label=%d pred=%d\n", prob, label, pred);
        }
    }

    printf("Inference time: %.6f seconds\n", elapsed_time);
    printf("Overall accuracy: %.2f%% (%d/%d correct)\n",
           (double)correct / NUM_SAMPLES * 100.0,
           correct,
           NUM_SAMPLES);
    freeCNN(model);
}
#endif

int main(void)
{
    test_sentence_logit_fp32_time();
    //sentence_all_f32_time();
    return 0;
}
