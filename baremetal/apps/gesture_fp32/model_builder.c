/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */
#include "model_builder.h"
#include "baremetal_timer.h"
#include "weights_fused_fp32.h"

CNN *build_gesture_cnn(ModelBuildStats *stats)
{
    timer_ticks_t def_start = timer_now();
    NNModule *transpose = nn_Transpose(50, 5, TRANSPOSE_CW_TO_WC, ELEM_FLOAT32);
    NNModule *conv1 = nn_Conv1d(5, 50, 3, 32, 1, 0, RELU, conv1d_1_weight, conv1d_1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv2 = nn_Conv1d(32, conv1->outputShape.W, 3, 64, 1, 0, RELU, conv1d_2_weight, conv1d_2_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv3 = nn_Conv1d(64, conv2->outputShape.W, 3, 128, 1, 0, RELU, conv1d_3_weight, conv1d_3_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv4 = nn_Conv1d(128, conv3->outputShape.W, 3, 256, 1, 0, RELU, conv1d_4_weight, conv1d_4_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *conv5 = nn_Conv1d(256, conv4->outputShape.W, 1, 256, 1, 0, RELU, conv1d_5_weight, conv1d_5_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *fc = nn_Linear(conv5->outputShape.C * conv5->outputShape.W, 4,
                             SOFTMAX, softmax_1_weight, softmax_1_bias,
                             NULL, NULL, ELEM_FLOAT32);
    timer_ticks_t def_end = timer_now();
    CNN *model = createCNN();
    addLayer(model, transpose);
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, conv4);
    addLayer(model, conv5);
    addLayer(model, fc);
    timer_ticks_t add_end = timer_now();
    if (stats) {
        stats->definition_seconds = timer_to_seconds(def_end - def_start);
        stats->addition_seconds = timer_to_seconds(add_end - def_end);
    }
    return model;
}
