/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef SENTENCE_INT8_MODEL_H
#define SENTENCE_INT8_MODEL_H

#include "nn_layer.h"
#include "weights_q.h"

static inline CNN *sentence_int8_create_model(void)
{
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
    NNModule *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, fc2_M,
                              fc2_zero_points, ELEM_INT8);

    CNN *model = createCNN();
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, maxpool);
    addLayer(model, fc1);
    addLayer(model, fc2);
    return model;
}

#endif
