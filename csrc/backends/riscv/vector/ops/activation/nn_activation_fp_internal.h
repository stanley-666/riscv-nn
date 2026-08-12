/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _NN_ACTIVATION_FP_H_
#define _NN_ACTIVATION_FP_H_

#include "nn_param.h"
#include <riscv_vector.h>

typedef void (*activate_store_kernel_f32_t)(float *dst, const float *src, int len);
typedef void (*activate_store_chunk_kernel_f32m2_t)(vfloat32m2_t vacc, float *output, size_t vl);
typedef void (*activate_store_chunk_kernel_f32m8_t)(vfloat32m8_t vacc, float *output, size_t vl);
typedef void (*activate_store_chunk_kernel_f32m4_t)(vfloat32m4_t vacc, float *output, size_t vl);

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act);

activate_store_kernel_f32_t select_activate_store_kernel_f32(ActivationType act);
activate_store_chunk_kernel_f32m2_t select_activate_store_chunk_kernel_f32m2(ActivationType act);
activate_store_chunk_kernel_f32m8_t select_activate_store_chunk_kernel_f32m8(ActivationType act);
activate_store_chunk_kernel_f32m4_t select_activate_store_chunk_kernel_f32m4(ActivationType act);
void activate_store_chunk_f32m2(ActivationType act, vfloat32m2_t vacc, float *output, size_t vl);
void activate_store_chunk_f32m8(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl);
void activate_store_chunk_f32m4(ActivationType act, vfloat32m4_t vacc, float *output, size_t vl);
#endif
