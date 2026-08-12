/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef _NN_OPS_VPU_CONV2D_H_
#define _NN_OPS_VPU_CONV2D_H_

#include "nn_param.h"

#define conv2d_int8_vpu conv2d_int8_vpu_m8

void conv2d_int8_vpu_m8(NNModule *layer, void *input, void *output);
void conv2d_int8_vpu_im2col_m8(NNModule *layer, void *input, void *output);
void conv2d_fp32_vpu(NNModule *layer, void *input, void *output);
void conv2d_fp32_vpu_im2col(NNModule *layer, void *input, void *output);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
