#ifndef _NN_OPS_VPU_CONV1D_I8_H_
#define _NN_OPS_VPU_CONV1D_I8_H_

#include "nn_param.h"

#define conv1d_i8_vpu conv1d_i8_vpu_m8

void conv1d_i8_vpu_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining2_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining4_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining8_m2(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining2_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining4_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining8_m4(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll2_acc8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll4_acc8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_im2col_unroll8_acc8_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining2_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining4_m8(NNModule *layer, void *input, void *output);
void conv1d_i8_vpu_chaining8_m8(NNModule *layer, void *input, void *output);
/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
