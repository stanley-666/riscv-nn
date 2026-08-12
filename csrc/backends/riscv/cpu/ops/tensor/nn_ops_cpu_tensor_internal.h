/* SPDX-FileContributor: Person: Stanley Lee */
/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_OPS_CPU_TENSOR_INTERNAL_H
#define NN_OPS_CPU_TENSOR_INTERNAL_H

#include "nn_param.h"

void transpose_cpu(NNModule *layer, void *input, void *output);
void save_cpu(NNModule *layer, void *input, void *output);
void add_cpu(NNModule *layer, void *input, void *output);

#endif

