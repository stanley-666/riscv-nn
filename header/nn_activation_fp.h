#ifndef _NN_ACTIVATION_FP_H_
#define _NN_ACTIVATION_FP_H_

#include "nn_param.h"
#include <riscv_vector.h>

typedef void (*activate_store_kernel_f32_t)(float *dst, const float *src, int len);
typedef void (*activate_store_chunk_kernel_f32_t)(vfloat32m8_t vacc, float *output, size_t vl);

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act);

activate_store_kernel_f32_t select_activate_store_kernel_f32(ActivationType act);
activate_store_chunk_kernel_f32_t select_activate_store_chunk_kernel_f32(ActivationType act);
void activate_store_chunk_f32(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl);

#endif
