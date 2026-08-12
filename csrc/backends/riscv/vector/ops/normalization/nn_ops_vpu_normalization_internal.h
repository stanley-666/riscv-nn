/* SPDX-FileContributor: Person: Stanley Lee */
/* Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef NN_OPS_VPU_NORMALIZATION_INTERNAL_H
#define NN_OPS_VPU_NORMALIZATION_INTERNAL_H

#include "nn_param.h"

void layernorm1d_fp32_vpu(NNModule *layer, void *input, void *output);

#endif
