/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _NN_ACTIVATION_INT_H_
#define _NN_ACTIVATION_INT_H_

#include "nn_param.h"
#include <riscv_vector.h>

typedef void (*requantize_store_chunk_i8_asym_per_channel_kernel_m8_t)(vint32m8_t vacc,
                                                                       const float *scale,
                                                                       const int32_t *zp,
                                                                       int8_t *dst,
                                                                       size_t vl);
typedef void (*requantize_store_chunk_i8_asym_per_channel_kernel_m4_t)(vint32m4_t vacc,
                                                                       const float *scale,
                                                                       const int32_t *zp,
                                                                       int8_t *dst,
                                                                       size_t vl);
typedef void (*requantize_store_chunk_i8_asym_per_channel_kernel_m2_t)(vint32m2_t vacc,
                                                                       const float *scale,
                                                                       const int32_t *zp,
                                                                       int8_t *dst,
                                                                       size_t vl);

requantize_store_chunk_i8_asym_per_channel_kernel_m8_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(ActivationType act);
requantize_store_chunk_i8_asym_per_channel_kernel_m4_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(ActivationType act);
requantize_store_chunk_i8_asym_per_channel_kernel_m2_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(ActivationType act);

#endif
