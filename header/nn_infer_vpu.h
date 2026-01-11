#ifndef _NN_INFER_VPU_H_
#define _NN_INFER_VPU_H_
// main forward logic
#include "nn_ops_vpu.h"

void forward_int8_vpu(CNN *net, void *input);
void forward_fp32_vpu(CNN *net, void *input);
#endif