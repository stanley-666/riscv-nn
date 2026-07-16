#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "nn_layer.h"
#include "nn_utils.h"
#if NN_BACKEND_CPU
#include "nn_infer_cpu.h"
#endif
#include "nn_infer_vpu.h"

#if NN_BACKEND_VECTOR
#define RUN_FORWARD(model, input) forward_fp32_vpu((model), (input))
#else
#define RUN_FORWARD(model, input) forward((model), (input))
#endif

#include "weights_fp32.h"
#include "sentence_model.h"
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
        SentenceFp32Layers layers = sentence_fp32_define_layers();
        clock_t layer_def_end = clock();
        double layer_def_elapsed = (double)(layer_def_end - layer_def_start) / CLOCKS_PER_SEC;
        printf("Layer definition time: %.6f seconds\n", layer_def_elapsed);
        printf("Creating CNN model...\n");
        CNN *model;

        printf("Adding layers to the model...\n");
        clock_t add_layer_start = clock();
        model = sentence_fp32_connect_layers(&layers);
        clock_t add_layer_end = clock();
        double add_layer_elapsed = (double)(add_layer_end - add_layer_start) / CLOCKS_PER_SEC;
        printf("Layer addition time: %.6f seconds\n", add_layer_elapsed);

        float embedding_f32[384];
        for (int i = 0; i < 384; ++i)
            embedding_f32[i] = (float)random_embedding[i];
        forward_input_bytes = 384 * sizeof(float);

        clock_t start_time = clock();
        NNInferenceProfile profile;
        uint64_t start_cycle = read_rdcycle();
        RUN_FORWARD(model, (void *)embedding_f32);
        uint64_t total_cycles = read_rdcycle() - start_cycle;
        clock_t end_time = clock();
#if NN_BACKEND_VECTOR
        forward_fp32_vpu_profile(model, (void *)embedding_f32, &profile);
#else
        profile.num_layers = 0;
#endif
        float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2); // 根據層數判斷最終輸出所在的 ping-pong buffer
        float probability = output[0];
#if NN_BACKEND_VECTOR
        /* Re-run outside the measured region with the final activation disabled
         * so the RVV approximation can be checked against scalar expf(). */
        layers.fc2->activation = NONE;
        forward_fp32_vpu(model, (void *)embedding_f32);
        float raw_logit = output[0];
        float reference_probability = activate_f32(raw_logit, SIGMOID);
        float approximation_error = fabsf(probability - reference_probability);
        layers.fc2->activation = SIGMOID;
#endif
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        for (int i = 0; i < profile.num_layers; ++i) {
            printf("Layer %d (%s): %lu cycles\n", i,
                   layer_type_name(profile.layers[i].type),
                   (unsigned long)profile.layers[i].cycles);
        }
        printf("Total inference: %lu cycles\n", (unsigned long)total_cycles);
        printf("Inference time: %.6f seconds\n", elapsed_time);
#if NN_BACKEND_VECTOR
        printf("Sigmoid raw logit: %.8f\n", raw_logit);
        printf("Sigmoid RVV approximation: %.8f\n", probability);
        printf("Sigmoid expf reference: %.8f\n", reference_probability);
        printf("Sigmoid absolute error: %.8f\n", approximation_error);
#endif

        // print output
        printf("Ground truth (valid_embedding): %d\n", valid_embedding);
        float prob = probability;
        int pred = prob >= 0.5f;
        printf("Model prediction (prob): %.4f -> label %d\n", prob, pred);

        if (pred == valid_embedding) {
            printf("Prediction correct!\n");
        } else {
            printf("Prediction incorrect!\n");
        }

        if (pred == 1)
            printf("Model predicts: Valid sentence\n");
        else
            printf("Model predicts: Invalid sentence\n");

        freeCNN(model);
    }

    void sentence_all_f32() {
        printf("1D CNN Inference Demo\n");
        /* definition layers  */
        printf("Initializing ping-pong buffers...\n");
        init_pingpong_buffer(SENTENCE_FP32_MAX_ELEMS, ELEM_FLOAT32);
        printf("Defining layers...\n");
        CNN *model = sentence_fp32_create_model();
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
            RUN_FORWARD(model, (void*)embedding_f32);
            clock_t end_time = clock();
            elapsed_time += (double)(end_time - start_time);
            // Six layers: conv1, conv2, conv3, maxpool, fc1, fc2.
            float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);

            float prob = output[0];
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
    return 0;
}
