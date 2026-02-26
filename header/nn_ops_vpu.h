#ifndef _NN_OPS_VPU_H_
#define _NN_OPS_VPU_H_

#include "nn_param.h"
#include "nn_activation_int.h"
#include "nn_activation_fp.h"
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>

void conv1d_i8_vpu(NNModule *layer, void *input, void *output);
void conv1d_fp32_vpu(NNModule *layer, void *input, void *output);

void conv2d_int8_vpu(NNModule *layer, void *input, void *output);
void conv2d_fp32_vpu(NNModule *layer, void *input, void *output);

void maxpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool1d_fp32_vpu(NNModule *layer, void *input, void *output);

void avgpool1d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_fp32_vpu(NNModule *layer, void *input, void *output);

void maxpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void maxpool2d_fp32_vpu(NNModule *layer, void *input, void *output);

void avgpool2d_int8_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_fp32_vpu(NNModule *layer, void *input, void *output);

void AdaptiveMaxPool1d_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_wc_fp32_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_int8_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_fp32_vpu(NNModule *layer, void *input, void *output);

void fullyconnected_int8_vpu(NNModule *layer, void *input, void *output);
void fullyconnected_fp32_vpu(NNModule *layer, void *input, void *output);
void transpose_vpu(NNModule *layer, void *input, void *output);
void save_vpu(NNModule *layer, void *input, void *output);
void add_vpu(NNModule *layer, void *input, void *output);
#endif
