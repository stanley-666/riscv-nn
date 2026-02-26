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
//#include "test_dataset_2658.h"
#include "test_dataset.h"

#define SENTENCE_FP32_MAX_ELEMS (384 * 256)  // max tensor elements across the model
static float buffer1_static[SENTENCE_FP32_MAX_ELEMS] __attribute__((aligned(64)));
static float buffer2_static[SENTENCE_FP32_MAX_ELEMS] __attribute__((aligned(64)));

uint64_t read_rdcycle() {
    uint64_t cycle;
    __asm__ volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

void init_pingpong_buffer(size_t num_elem, elem_type dtype) {
    (void)num_elem;
    (void)dtype;
    size_t size = SENTENCE_FP32_MAX_ELEMS * sizeof(float);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

#ifdef WEIGHTS_FP32_H
    void test_sentence_logit_fp32() {
        printf("1D CNN Inference Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_FP32_MAX_ELEMS, ELEM_FLOAT32);
        printf("Defining layers...\n");
        clock_t layer_def_start = clock();
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *transposed_conv3 = nn_Transpose(conv3->outputShape.W, conv3->outputShape.C, TRANSPOSE_WC_TO_CW, ELEM_FLOAT32);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1,  ELEM_FLOAT32);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, NULL, NULL, ELEM_FLOAT32);
        clock_t layer_def_end = clock();
        double layer_def_elapsed = (double)(layer_def_end - layer_def_start) / CLOCKS_PER_SEC;
        printf("Layer definition time: %.6f seconds\n", layer_def_elapsed);
        printf("Creating CNN model...\n");
        CNN* model = createCNN();

        printf("Adding layers to the model...\n");
        clock_t add_layer_start = clock();
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, transposed_conv3);
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        clock_t add_layer_end = clock();
        double add_layer_elapsed = (double)(add_layer_end - add_layer_start) / CLOCKS_PER_SEC;
        printf("Layer addition time: %.6f seconds\n", add_layer_elapsed);

        float embedding_f32[384];
        for (int i = 0; i < 384; ++i)
            embedding_f32[i] = (float)random_embedding[i];
        forward_input_bytes = 384 * sizeof(float);

        clock_t start_time = clock();
        uint64_t start_cycle = read_rdcycle();
        forward_fp32_vpu(model, (void*)embedding_f32);
        uint64_t end_cycle = read_rdcycle();
        clock_t end_time = clock();
        float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2); // 根據層數判斷最終輸出所在的 ping-pong buffer
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
         printf("Inference time: %lu cycles\n", end_cycle - start_cycle);
        printf("Inference time: %.6f seconds\n", elapsed_time);

        // print output
        printf("Ground truth (valid_embedding): %d\n", valid_embedding);
        float prob = output[0];
        int pred = prob >= 0.5f;
        printf("Model prediction (prob): %.4f -> label %d\n", prob, pred);

        if (pred == valid_embedding) {
            printf("Prediction correct!\n");
        } else {
            printf("Prediction incorrect!\n");
        }

        if (pred == 1)
            printf("→ Model predicts: Valid sentence\n");
        else
            printf("→ Model predicts: Invalid sentence\n");

        freeCNN(model);
    }

    void sentence_all_f32() {
        printf("1D CNN Inference Demo\n");
        /* definition layers  */
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_FP32_MAX_ELEMS, ELEM_FLOAT32);
        printf("Defining layers...\n");
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *transposed_conv3 = nn_Transpose(conv3->outputShape.W, conv3->outputShape.C, TRANSPOSE_WC_TO_CW, ELEM_FLOAT32);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1,  ELEM_FLOAT32);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, NULL, NULL, ELEM_FLOAT32);
        NNModule *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, NULL, NULL, ELEM_FLOAT32);

        /* connect layers together */
        CNN* model = createCNN();
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, transposed_conv3);
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        printf("Completed model creation.\n");
  
        /* inference test data*/
        int correct = 0 ;
        double elapsed_time = 0;
        float embedding_f32[384];
        forward_input_bytes = 384 * sizeof(float);
        for (int i = 0; i < NUM_SAMPLES; i++) {
            const int8_t *embedding = test_embeddings[i];
            bool label = test_labels[i];

            for (int j = 0; j < 384; ++j)
                embedding_f32[j] = (float)embedding[j];

            clock_t start_time = clock();
            forward_fp32_vpu(model, (void*)embedding_f32);
            clock_t end_time = clock();
            elapsed_time += (double)(end_time - start_time);
            // layer count: conv1,conv2,conv3,transpose,maxpool,fc1,fc2 => 7 (odd), 最終輸出在 buffer2
            float *out_ptr = (float *)buffer2;
            float prob = out_ptr[0];
            int pred = (prob >= 0.5f) ? 1 : 0;

            if (pred == label) {
                correct++;
            } else {
                printf("[%d] ", i);
                for(int j = 0; j < 384; j++)
                    printf("%d ", embedding[j]);
                printf("\n");

                printf("prob=%.4f label=%d pred=%d\n", prob, label, pred);
            }
        }

        elapsed_time /= CLOCKS_PER_SEC;
        printf("Inference time: %.6f seconds\n", elapsed_time);
        printf("Overall accuracy: %.2f%% (%d/%d correct)\n", (double)correct / NUM_SAMPLES * 100.0, correct, NUM_SAMPLES);
        freeCNN(model);
    }
#endif

int main() {
    test_sentence_logit_fp32();
    sentence_all_f32();
    return 0;
}
