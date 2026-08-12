/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef _NN_INFER_VPU_H_
#define _NN_INFER_VPU_H_
// main forward logic
#include "nn_param.h"

#define NN_INFERENCE_PROFILE_MAX_LAYERS 64

typedef struct {
    LayerType type;
    uint64_t cycles;
} NNLayerCycleProfile;

typedef struct {
    uint64_t total_cycles;
    int num_layers;
    NNLayerCycleProfile layers[NN_INFERENCE_PROFILE_MAX_LAYERS];
} NNInferenceProfile;

void forward_int8_vpu(CNN *net, void *input);
void forward_fp32_vpu(CNN *net, void *input);
void forward_fp32_vpu_profile(CNN *net, void *input, NNInferenceProfile *profile);
#endif
