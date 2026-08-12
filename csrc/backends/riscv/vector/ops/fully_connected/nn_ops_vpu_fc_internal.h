/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef _NN_OPS_VPU_FC_H_
#define _NN_OPS_VPU_FC_H_

#include "nn_param.h"

#define fullyconnected_int8_vpu fullyconnected_int8_vpu_m8
#define fullyconnected_fp32_vpu fullyconnected_fp32_vpu_m8

void fullyconnected_int8_vpu_m8(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining2_m8(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining4_m8(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining8_m8(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_m4(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining2_m4(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining4_m4(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining8_m4(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_m2(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining2_m2(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining4_m2(NNModule *layer, void *input, void *output);
void fullyconnected_int8_vpu_chaining8_m2(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_m8(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll2_m8(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll4_m8(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll8_m8(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_m4(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll2_m4(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll4_m4(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll8_m4(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_m2(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll2_m2(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll4_m2(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_unroll8_m2(NNModule *layer, void *input, void *output);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
