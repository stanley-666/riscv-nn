#ifndef _NN_ACTIVATION_INT_H_
#define _NN_ACTIVATION_INT_H_

#include "nn_param.h"
#include <riscv_vector.h>

typedef void (*requantize_store_kernel_t)(const int32_t *src,
                                          const float *scale,
                                          const int32_t *zp,
                                          int8_t *dst,
                                          int len);

requantize_store_kernel_t select_requantize_store_kernel(ActivationType act);
void requantize_activate_store_rvv(const int32_t *src,
                                   const float *scale,
                                   const int32_t *zp,
                                   int8_t *dst,
                                   int len,
                                   ActivationType act);

#endif
