#ifndef _NN_ACTIVATION_FP_H_
#define _NN_ACTIVATION_FP_H_

#include "nn_param.h"
#include <riscv_vector.h>

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act);

void activate_store_chunk_f32(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl);
#endif
