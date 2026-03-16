#ifndef _NN_ACTIVATION_INT_H_
#define _NN_ACTIVATION_INT_H_

#include "nn_param.h"
#include <riscv_vector.h>

typedef void (*requantize_store_chunk_i8_asym_per_channel_kernel_t)(vint32m8_t vacc,
                                                                    const float *scale,
                                                                    const int32_t *zp,
                                                                    int8_t *dst,
                                                                    size_t vl);

requantize_store_chunk_i8_asym_per_channel_kernel_t
select_requantize_store_chunk_i8_asym_per_channel_kernel(ActivationType act);

#endif
