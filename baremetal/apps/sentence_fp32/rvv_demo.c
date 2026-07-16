#include "cnn_demo.h"
#include <stdbool.h>
#include <stdio.h>

#if ENABLE_VECTOR
#include "baremetal_timer.h"
#include "model_builder.h"
#include "nn_infer_vpu.h"
#include "nn_utils.h"
#include "random_embedding.h"
#include "test_dataset.h"

void init_pingpong_buffer(size_t num_elem, elem_type dtype)
{
    size_t bytes = num_elem * sizeof_dtype(dtype);
    if (buffer1)
        safe_free(buffer1);
    if (buffer2)
        safe_free(buffer2);
    buffer1 = safe_calloc(num_elem, sizeof_dtype(dtype));
    buffer2 = safe_calloc(num_elem, sizeof_dtype(dtype));
    printf("ping-pong buffers allocated: %lu bytes each\n", (unsigned long)bytes);
}

static void log_build_stats(const ModelBuildStats *stats) {
    if (!stats)
        return;
    printf("NNModule definition time: %.6f seconds\n", stats->definition_seconds);
    printf("NNModule addition time : %.6f seconds\n", stats->addition_seconds);
}

static void release_resources(CNN *model) {
    freeCNN(model);
    free_pingpong_buffer();
}

void run_rvv_single_demo() {
    ModelBuildStats stats = {0};
    CNN *model = build_sentence_cnn(&stats);
    log_build_stats(&stats);
    init_pingpong_buffer(1024*1024, ELEM_FLOAT32);

    float embedding_f32[384];
    for (int i = 0; i < 384; ++i)
        embedding_f32[i] = (float)random_embedding[i];
    forward_input_bytes = 384 * sizeof(float);

    timer_ticks_t total_start = timer_now();
    forward_fp32_vpu(model, (void *)embedding_f32);
    timer_ticks_t total_cycles = timer_now() - total_start;

    NNInferenceProfile profile;
    forward_fp32_vpu_profile(model, (void *)embedding_f32, &profile);
    float *output = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);

    for (int i = 0; i < profile.num_layers; ++i) {
        printf("Layer %d (%s): %lu cycles\n", i,
               layer_type_name(profile.layers[i].type),
               (unsigned long)profile.layers[i].cycles);
    }
    printf("Total inference: %lu cycles\n", (unsigned long)total_cycles);
    printf("Inference time: %.6f seconds\n", timer_to_seconds(total_cycles));
    
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
    release_resources(model);
}

void run_rvv_dataset_demo() {

    ModelBuildStats stats = {0};
    CNN *model = build_sentence_cnn(&stats);
    log_build_stats(&stats);
    init_pingpong_buffer(1024*1024, ELEM_FLOAT32);

    int correct = 0;
    float embedding_f32[384];
    forward_input_bytes = 384 * sizeof(float);
    timer_ticks_t elapsed_time = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        const int8_t *embedding = test_embeddings[i];
        bool label = test_labels[i];

        for (int j = 0; j < 384; ++j)
            embedding_f32[j] = (float)embedding[j];

        timer_ticks_t start_time = timer_now();
        forward_fp32_vpu(model, (void*)embedding_f32);
        timer_ticks_t end_time = timer_now();
        elapsed_time += (end_time - start_time);

        float *out_ptr = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
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

    printf("Inference time: %.6f seconds\n", timer_to_seconds(elapsed_time));
    printf("Inference cycle: %ld cycles\n", elapsed_time);
    printf("Overall accuracy: %.2f%% (%d/%d correct)\n", (double)correct / NUM_SAMPLES * 100.0, correct, NUM_SAMPLES);
    release_resources(model);
}

#else

void run_rvv_single_demo() {
    printf("[RVV] support not built into this binary.\n");
}

void run_rvv_dataset_demo() {
    printf("[RVV] dataset sweep unavailable in scalar-only builds.\n");
}

#endif
