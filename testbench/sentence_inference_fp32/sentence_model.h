/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENTENCE_FP32_MODEL_H
#define SENTENCE_FP32_MODEL_H

#include "nn_layer.h"
#include "weights_fp32.h"

typedef struct {
    NNModule *conv1;
    NNModule *conv2;
    NNModule *conv3;
    NNModule *maxpool;
    NNModule *fc1;
    NNModule *fc2;
} SentenceFp32Layers;

static inline SentenceFp32Layers sentence_fp32_define_layers(void)
{
    SentenceFp32Layers layers;
    layers.conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU,
                             conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
    layers.conv2 = nn_Conv1d(64, layers.conv1->outputShape.W, 5, 128, 1, 2, RELU,
                             conv2_weight, conv2_bias, NULL, NULL, ELEM_FLOAT32);
    layers.conv3 = nn_Conv1d(128, layers.conv2->outputShape.W, 3, 256, 1, 1, RELU,
                             conv3_weight, conv3_bias, NULL, NULL, ELEM_FLOAT32);
    layers.maxpool = nn_AdaptiveMaxPool1d(layers.conv3->outputShape.C,
                                          layers.conv3->outputShape.W, 1,
                                          ELEM_FLOAT32);
    layers.fc1 = nn_Linear(layers.maxpool->outputShape.C * layers.maxpool->outputShape.W,
                           128, RELU, fc1_weight, fc1_bias, NULL, NULL, ELEM_FLOAT32);
    layers.fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias,
                           NULL, NULL, ELEM_FLOAT32);
    return layers;
}

static inline CNN *sentence_fp32_connect_layers(const SentenceFp32Layers *layers)
{
    CNN *model = createCNN();
    addLayer(model, layers->conv1);
    addLayer(model, layers->conv2);
    addLayer(model, layers->conv3);
    addLayer(model, layers->maxpool);
    addLayer(model, layers->fc1);
    addLayer(model, layers->fc2);
    return model;
}

static inline CNN *sentence_fp32_create_model(void)
{
    SentenceFp32Layers layers = sentence_fp32_define_layers();
    return sentence_fp32_connect_layers(&layers);
}

#endif
