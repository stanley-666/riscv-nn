#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

#if NN_BACKEND_VECTOR
#define RUN_FORWARD(model, input) forward_int8_vpu((model), (input))
#else
#define RUN_FORWARD(model, input) forward((model), (input))
#endif

#include "weights_q.h"
#include "sentence_model.h"
#include "test_dataset.h"

#define SENTENCE_INT8_MAX_ELEMS (384 * 256)  // max tensor elements across the model
static int8_t buffer1_static[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));
static int8_t buffer2_static[SENTENCE_INT8_MAX_ELEMS] __attribute__((aligned(64)));
void init_pingpong_buffer(size_t num_elem, elem_type dtype) {
    (void)num_elem;
    (void)dtype;
    size_t size = SENTENCE_INT8_MAX_ELEMS * sizeof(int8_t);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

static void sentence_all_i8_shared(void)
{
    printf("1D CNN Inference INT8 Demo\n");
    printf("Weights/model definition: shared CPU/RVV path\n");
    printf("Initializing ping-pong buffers...\n");
    init_pingpong_buffer(SENTENCE_INT8_MAX_ELEMS, ELEM_INT8);
    printf("Defining layers...\n");
    CNN *model = sentence_int8_create_model();
    printf("Completed model creation.\n");

    int correct = 0;
    double elapsed_cycles = 0.0;
    forward_input_bytes = 384 * sizeof(int8_t);
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        const int8_t *embedding = test_embeddings[i];
        const bool label = test_labels[i];
        const uint64_t start = nn_runtime_read_cycles();
        RUN_FORWARD(model, (void *)embedding);
        const uint64_t end = nn_runtime_read_cycles();
        elapsed_cycles += (double)(end - start);

        const int8_t *output = (const int8_t *)
            ((model->numModules % 2 == 0) ? buffer1 : buffer2);
        printf("Output[%d]: %d, Label: %d\n", i, output[0], label);
        if (output[0] == label) {
            printf("[%d] correct\n", i);
            ++correct;
        } else {
            printf("[%d] false\n", i);
        }
    }

    printf("Inference time: %.6f seconds\n",
           elapsed_cycles / (double)nn_runtime_cycle_frequency_hz());
    printf("Overall accuracy: %.2f%% (%d/%d correct)\n",
           (double)correct / NUM_SAMPLES * 100.0, correct, NUM_SAMPLES);
    freeCNN(model);
}

int main() {
    sentence_all_i8_shared();
    return 0;
}
