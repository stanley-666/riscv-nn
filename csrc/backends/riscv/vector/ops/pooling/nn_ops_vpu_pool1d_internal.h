/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef _NN_OPS_VPU_POOL1D_H_
#define _NN_OPS_VPU_POOL1D_H_

#include "nn_param.h"

void maxpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool1d_fp32_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_fp32_vpu(NNModule *layer, void *input, void *output);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
