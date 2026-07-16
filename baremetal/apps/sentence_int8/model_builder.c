/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "model_builder.h"

#include "baremetal_timer.h"
#include "weights_q.h"

CNN *build_sentence_int8_cnn(ModelBuildStats *stats)
{
    timer_ticks_t definition_start = timer_now();
    NNModule *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU,
                                conv1_weight, conv1_bias, conv1_M,
                                conv1_zero_points, ELEM_INT8);
    NNModule *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU,
                                conv2_weight, conv2_bias, conv2_M,
                                conv2_zero_points, ELEM_INT8);
    NNModule *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU,
                                conv3_weight, conv3_bias, conv3_M,
                                conv3_zero_points, ELEM_INT8);
    NNModule *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C,
                                             conv3->outputShape.W, 1,
                                             ELEM_INT8);
    NNModule *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W,
                              128, RELU, fc1_weight, fc1_bias, fc1_M,
                              fc1_zero_points, ELEM_INT8);
    NNModule *fc2 = nn_Linear(128, 1, NONE, fc2_weight, fc2_bias, fc2_M,
                              fc2_zero_points, ELEM_INT8);
    timer_ticks_t definition_end = timer_now();

    CNN *model = createCNN();
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, maxpool);
    addLayer(model, fc1);
    addLayer(model, fc2);
    timer_ticks_t addition_end = timer_now();

    if (stats) {
        stats->definition_seconds = timer_to_seconds(definition_end - definition_start);
        stats->addition_seconds = timer_to_seconds(addition_end - definition_end);
    }
    return model;
}
