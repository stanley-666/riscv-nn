#ifndef _NN_ACTIVATION_FP_H_
#define _NN_ACTIVATION_FP_H_

#include "nn_param.h"

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act);

#endif
