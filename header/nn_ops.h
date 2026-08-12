/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef RISCV_NN_OPS_H
#define RISCV_NN_OPS_H

#include <stddef.h>
#include "nn_param.h"

/* Stable operator entry points used by the inference dispatcher. */
void conv1d_i8_vpu_im2col_m8(NNModule *layer, void *input, void *output);
void conv1d_fp32_vpu_im2col_m8(NNModule *layer, void *input, void *output);
void conv2d_int8_vpu_im2col_m8(NNModule *layer, void *input, void *output);
void conv2d_fp32_vpu_im2col(NNModule *layer, void *input, void *output);

void fullyconnected_int8_vpu_m8(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu_m8(NNModule *layer, void *input, void *output);
#define fullyconnected_int8_vpu fullyconnected_int8_vpu_m8
#define fullyconnected_fp32_vpu fullyconnected_fp32_vpu_m8

void maxpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool1d_fp32_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_fp32_vpu(NNModule *layer, void *input, void *output);

void maxpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool2d_fp32_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_fp32_vpu(NNModule *layer, void *input, void *output);

void transpose_vpu(NNModule *layer, void *input, void *output);
void save_vpu(NNModule *layer, void *input, void *output);
void add_vpu(NNModule *layer, void *input, void *output);
void layernorm1d_fp32_vpu(NNModule *layer, void *input, void *output);
void attention1d_fp32_vpu(NNModule *layer, void *input, void *output);

/* Full-tensor RVV activation operators. */
void softmax_f32_rvv(const float *input, float *output, size_t length);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

#endif
