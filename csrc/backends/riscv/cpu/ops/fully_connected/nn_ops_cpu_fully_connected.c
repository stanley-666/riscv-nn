/* SPDX-FileContributor: Person: Stanley Lee */
/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_ops_cpu_fully_connected_internal.h"
#include "nn_utils.h"
#include <math.h>
#include <string.h>

void fullyconnected_cpu(NNModule *layer, void *input, void *output)
{
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    
    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *input_f32 = (float*)input;
        float *output_f32 = (float*)output;
        const float *weights_f32 = (const float*)layer->params.fc.weights;
        const float *bias_f32 = (const float*)layer->params.fc.bias;
        ActivationType act = layer->activation;

        if (act == SOFTMAX) {
            printf("Softmax activation in FC layer\n");
            for (int o = 0; o < outW; o++) {
                float sum = bias_f32[o];
                for (int i = 0; i < inDim; i++)
                    sum += input_f32[i] * weights_f32[o * inDim + i];
                output_f32[o] = sum; // logits
            }
            softmax_f32(output_f32, output_f32, outW);
        } else {
            for (int o = 0; o < outW; o++) {
                float sum = bias_f32[o];
                for (int i = 0; i < inDim; i++)
                    sum += input_f32[i] * weights_f32[o * inDim + i];
                output_f32[o] = activate_f32(sum, act);
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *input_i8 = (int8_t*)input;
        int8_t *output_i8 = (int8_t*)output;
        const int8_t *weights_i8 = (const int8_t*)layer->params.fc.weights;
        const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
        const float *M = (const float*)layer->params.fc.M;
        const int32_t *Z = (const int32_t*)layer->params.fc.zps;

        for (int o = 0; o < outW; o++) {
            int32_t sum = bias_i32[o];
            for (int i = 0; i < inDim; i++) {
                sum += input_i8[i] * weights_i8[o * inDim + i];
            }
            int8_t requantized = requantize_int8_asymmetric(sum, M[o], Z[o]);
            output_i8[o] = activate_i8(requantized, layer->activation);
        }
        break;
    }
    default:
        printf("Unsupported dtype in fc_forward\n");
        exit(EXIT_FAILURE);
    }
}


