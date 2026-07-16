/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_tensor_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

void transpose_cpu(NNModule *layer, void *input, void *output)
{
    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    TransposeType mode = layer->params.transpose.mode;

    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *out = (float *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c)
                for (int w = 0; w < width; ++w)
                    out[w * channels + c] = in[c * width + w];
        } else {
            for (int w = 0; w < width; ++w)
                for (int c = 0; c < channels; ++c)
                    out[c * width + w] = in[w * channels + c];
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c)
                for (int w = 0; w < width; ++w)
                    out[w * channels + c] = in[c * width + w];
        } else {
            for (int w = 0; w < width; ++w)
                for (int c = 0; c < channels; ++c)
                    out[c * width + w] = in[w * channels + c];
        }
        break;
    }
    default:
        printf("Unsupported dtype in transpose_forward\n");
        exit(EXIT_FAILURE);
    }
}

void save_cpu(NNModule *layer, void *input, void *output)
{
    if (!layer->params.save.buffer || layer->params.save.bytes == 0) {
        printf("Save layer buffer not initialized (set buffer externally)\n");
        exit(EXIT_FAILURE);
    }
    memcpy(layer->params.save.buffer, input, layer->params.save.bytes);
    memcpy(output, input, layer->params.save.bytes);
}

void add_cpu(NNModule *layer, void *input, void *output)
{
    if (!layer->params.add.skip || layer->params.add.bytes == 0) {
        printf("Add layer skip buffer not initialized\n");
        exit(EXIT_FAILURE);
    }
    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *skip = (float *)layer->params.add.skip;
        float *out = (float *)output;
        int len = (int)(layer->params.add.bytes / sizeof(float));
        for (int i = 0; i < len; ++i)
            out[i] = activate_f32(in[i] + skip[i], layer->activation);
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *skip = (int8_t *)layer->params.add.skip;
        int8_t *out = (int8_t *)output;
        int len = (int)(layer->params.add.bytes / sizeof(int8_t));
        for (int i = 0; i < len; ++i)
            out[i] = activate_i8((int8_t)(in[i] + skip[i]), layer->activation);
        break;
    }
    default:
        printf("Unsupported dtype in add_cpu\n");
        exit(EXIT_FAILURE);
    }
}

