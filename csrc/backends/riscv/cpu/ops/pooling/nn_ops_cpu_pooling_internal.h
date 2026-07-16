/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_OPS_CPU_POOLING_INTERNAL_H
#define NN_OPS_CPU_POOLING_INTERNAL_H

#include "nn_param.h"

void pool1d_cpu(NNModule *layer, void *input, void *output);
void pool2d_cpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_cpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_cpu(NNModule *layer, void *input, void *output);

#endif

