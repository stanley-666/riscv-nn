/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */
#ifndef GESTURE_MODEL_BUILDER_H
#define GESTURE_MODEL_BUILDER_H
#include "nn_layer.h"
typedef struct {
    double definition_seconds;
    double addition_seconds;
} ModelBuildStats;
CNN *build_gesture_cnn(ModelBuildStats *stats);
#endif
