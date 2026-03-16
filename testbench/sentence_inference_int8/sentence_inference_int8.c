#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include "nn_layer.h"
#include "nn_utils.h"
#include "nn_infer_cpu.h"
#include "nn_infer_vpu.h"
#include "nn_ops_vpu.h"

#include "weights_q.h"
#include "random_embedding.h"
//#include "test_dataset_2658.h"
#include "test_dataset.h"

#define SENTENCE_INT8_MAX_ELEMS (384 * 256)  // max tensor elements across the model
static int8_t buffer1_static[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));
static int8_t buffer2_static[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));
uint64_t read_rdcycle() {
    uint64_t cycle;
    __asm__ volatile ("rdcycle %0" : "=r" (cycle));
    return cycle;
}

void init_pingpong_buffer(size_t num_elem, elem_type dtype) {
    (void)num_elem;
    (void)dtype;
    size_t size = SENTENCE_INT8_MAX_ELEMS * sizeof(int8_t);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

#ifdef WEIGHTS_Q_H
    void sentence_1dcnn_i8() {
        printf("1D CNN Inference Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);
        printf("Defining layers...\n");
        clock_t layer_def_start = clock();
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
        NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);
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
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        clock_t add_layer_end = clock();
        double add_layer_elapsed = (double)(add_layer_end - add_layer_start) / CLOCKS_PER_SEC;
        printf("Layer addition time: %.6f seconds\n", add_layer_elapsed);
        /* input data */
        int8_t *embedding = (int8_t*)random_embedding;
        forward_input_bytes = 384 * sizeof(int8_t);
        // inference
        clock_t start_time = clock();
        forward(model, embedding);
        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Inference time: %.6f seconds\n", elapsed_time);

        // print output
        const int8_t *output = (int8_t *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
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

    void sentence_all_i8() {
        printf("1D CNN Inference INT8 Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);
        printf("Defining layers...\n");
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
        NNModule *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);

        CNN* model = createCNN();
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        printf("Completed model creation.\n");

        int correct = 0 ;
        forward_input_bytes = 384 * sizeof(int8_t);
        clock_t start_time = clock();
        for (int i = 0; i < NUM_SAMPLES; i++) {
            const int8_t *embedding = test_embeddings[i];
            bool label = test_labels[i];

            forward(model, (void *)embedding);
            int8_t *output = (int8_t *) ((model->numModules % 2 == 0) ? buffer1 : buffer2);  // 最後輸出在 buffer1


            if (output[0] == label) {
                printf("[%d] correct\n", i);
                correct++;
            } else {
                printf("[%d] false", i);
                /*
                for(int j = 0; j < 384; j++)
                    printf("%d ", embedding[j]);
                */
                printf("\n");
            }
        }

        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        printf("Inference time: %.6f seconds\n", elapsed_time);
        printf("Overall accuracy: %.2f%% (%d/%d correct)\n", (double)correct / NUM_SAMPLES * 100.0, correct, NUM_SAMPLES);
        freeCNN(model);
        
    }

    void sentence_i8_rvv() {
        printf("1D CNN Inference RVV INT8 Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);
        printf("Defining layers...\n");
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, RELU, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
        NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);

        CNN* model = createCNN();
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        printf("Completed model creation.\n");
        forward_input_bytes = 384 * sizeof(int8_t);
        int correct = 0 ;
        const int8_t *embedding = random_embedding;
        bool label = valid_embedding;
        clock_t start_time = clock();
        uint64_t start_cycle = read_rdcycle();
        forward_int8_vpu(model, (void *)embedding);
        uint64_t end_cycle = read_rdcycle();
        clock_t end_time = clock();
        const int8_t *output = (int8_t *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
        printf("Model prediction (output[0]): %d\n", output[0]);
        if (output[0] == label) {
            printf("correct\n");
            correct++;
        } else {
            printf("false\n");
        }

        uint64_t cycle_diff = end_cycle - start_cycle;
        double elapsed_time = (double)(end_time - start_time);
        printf("Inference cycles: %lu cycles\n", cycle_diff);
        elapsed_time = elapsed_time / CLOCKS_PER_SEC;
        printf("Inference time: %.6f seconds\n", elapsed_time);
        freeCNN(model);
    }

    void sentence_all_i8_rvv() {
        printf("1D CNN Inference RVV INT8 Demo\n");
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);
        printf("Defining layers...\n");
        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
        NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, NONE, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
        NNModule *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);

        CNN* model = createCNN();
        addLayer(model, conv1);
        addLayer(model, conv2);
        addLayer(model, conv3);
        addLayer(model, maxpool);
        addLayer(model, fc1);
        addLayer(model, fc2);
        printf("Completed model creation.\n");
        forward_input_bytes = 384 * sizeof(int8_t);
        int correct = 0 ;
        double elapsed_time = 0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            const int8_t *embedding = test_embeddings[i];
            bool label = test_labels[i];
            clock_t start_time = clock();
            forward_int8_vpu(model, (void *)embedding);
            clock_t end_time = clock();
            elapsed_time += (double)(end_time - start_time);
            const int8_t *output = (int8_t *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
            printf("Output[%d]: %d, Label: %d\n", i, output[0], label);
            if (output[0] == label) {
                printf("[%d] correct\n", i);
                correct++;
            } else {
                printf("[%d] false", i);
                /*
                for(int j = 0; j < 384; j++)
                    printf("%d ", embedding[j]);
                */
                printf("\n");
            }
        }

        elapsed_time = elapsed_time / CLOCKS_PER_SEC;
        printf("Inference time: %.6f seconds\n", elapsed_time);
        printf("Overall accuracy: %.2f%% (%d/%d correct)\n", (double)correct / NUM_SAMPLES * 100.0, correct, NUM_SAMPLES);
        freeCNN(model);
        
    }

    // Conv output stays NHWC/WC, then use WC-optimized adaptive maxpool1d directly.
    void sentence_i8_rvv_wc_pool_direct() {
        printf("1D CNN RVV INT8 Demo (WC AdaptivePool direct)\n");
        init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);

        NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
        NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
        NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
        NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
        NNModule *fc1 = nn_Linear(conv3->outputShape.C, 128, RELU, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
        NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);

        const int8_t *embedding = random_embedding;
        bool label = valid_embedding;

        uint64_t start_cycle = read_rdcycle();
        conv1d_i8_vpu(conv1, (void *)embedding, buffer1);
        conv1d_i8_vpu(conv2, buffer1, buffer2);
        conv1d_i8_vpu(conv3, buffer2, buffer1);
        AdaptiveMaxPool1d_wc_int8_vpu(maxpool, buffer1, buffer2);
        fullyconnected_int8_vpu(fc1, buffer2, buffer1);
        fullyconnected_int8_vpu(fc2, buffer1, buffer2);
        uint64_t end_cycle = read_rdcycle();

        const int8_t *output = (const int8_t *)buffer2;
        printf("Inference cycles: %lu cycles\n", (unsigned long)(end_cycle - start_cycle));
        printf("Model prediction (output[0]): %d, label: %d\n", output[0], label);

    }
#endif

int main() {
    //sentence_i8_rvv();
    sentence_all_i8_rvv();
    return 0;
}
