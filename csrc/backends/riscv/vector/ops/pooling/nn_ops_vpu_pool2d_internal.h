#ifndef _NN_OPS_VPU_POOL2D_H_
#define _NN_OPS_VPU_POOL2D_H_

#include "nn_param.h"

void maxpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool2d_fp32_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_fp32_vpu(NNModule *layer, void *input, void *output);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
