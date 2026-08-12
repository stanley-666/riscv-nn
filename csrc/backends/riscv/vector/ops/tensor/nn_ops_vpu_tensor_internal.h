/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef _NN_OPS_VPU_TENSOR_H_
#define _NN_OPS_VPU_TENSOR_H_

#include "nn_param.h"

void transpose_vpu(NNModule *layer, void *input, void *output);
void save_vpu(NNModule *layer, void *input, void *output);
void add_vpu(NNModule *layer, void *input, void *output);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
