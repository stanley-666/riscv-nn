#ifndef _NN_OPS_VPU_H_
#define _NN_OPS_VPU_H_

#include "nn_param.h"
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
/* RVV use NHWC layout */
void requantize_activate_store_rvv(const int32_t *src, const int32_t *bias, const float *scale, const int32_t *zp, int8_t *dst, int len, ActivationType act);
void activate_store_rvv_f32(float *dst, const float *src, int len, ActivationType act);
void conv1d_i8_vpu(NNModule *layer, void *input, void *output);
void conv1d_fp32_vpu(NNModule *layer, void *input, void *output);
void conv2d_int8_vpu(NNModule *layer, void *input, void *output);
void conv2d_fp32_vpu(NNModule *layer, void *input, void *output);
void maxpool1d_vpu(NNModule *layer, void *input, void *output);
void avgpool1d_vpu(NNModule *layer, void *input, void *output);
void maxpool2d_vpu(NNModule *layer, void *input, void *output);
void avgpool2d_vpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_vpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_vpu(NNModule *layer, void *input, void *output);
void fullyconnected_vpu(NNModule *layer, void *input, void *output);
void transpose_vpu(NNModule *layer, void *input, void *output);
void save_vpu(NNModule *layer, void *input, void *output);
void add_vpu(NNModule *layer, void *input, void *output);
#endif
