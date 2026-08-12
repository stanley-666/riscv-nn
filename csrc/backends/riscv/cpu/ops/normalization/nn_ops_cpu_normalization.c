/* SPDX-FileContributor: Person: Stanley Lee */
/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */

#include "nn_ops_cpu_normalization_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void layernorm1d_cpu(NNModule *layer, void *input, void *output)
{
    if (layer->dtype != ELEM_FLOAT32) {
        printf("Unsupported dtype in layernorm1d_cpu\n");
        exit(EXIT_FAILURE);
    }

    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    const float *in = (const float *)input;
    float *out = (float *)output;
    const float *weight = (const float *)layer->params.layernorm.weight;
    const float *bias = (const float *)layer->params.layernorm.bias;
    const float eps = layer->params.layernorm.eps;

    for (int w = 0; w < width; ++w) {
        const float *row = &in[w * channels];
        float *row_out = &out[w * channels];
        double mean = 0.0;
        for (int c = 0; c < channels; ++c)
            mean += row[c];
        mean /= (double)channels;

        double var = 0.0;
        for (int c = 0; c < channels; ++c) {
            double diff = (double)row[c] - mean;
            var += diff * diff;
        }
        var /= (double)channels;
        float inv_std = 1.0f / sqrtf((float)var + eps);

        for (int c = 0; c < channels; ++c) {
            float norm = (row[c] - (float)mean) * inv_std;
            row_out[c] = norm * weight[c] + bias[c];
        }
    }
}
