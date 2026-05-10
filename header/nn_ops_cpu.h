#ifndef _NN_OPS_CPU_H_
#define _NN_OPS_CPU_H_

#include "nn_param.h"

void conv1d_cpu(NNModule *layer, void *input, void *output);
void conv2d_cpu(NNModule *layer, void *input, void *output);
void pool1d_cpu(NNModule *layer, void *input, void *output);
void pool2d_cpu(NNModule *layer, void *input, void *output);
void fullyconnected_cpu(NNModule *layer, void *input, void *output);
void layernorm1d_cpu(NNModule *layer, void *input, void *output);
void attention1d_cpu(NNModule *layer, void *input, void *output);
void AdaptiveMaxPool1d_cpu(NNModule *layer, void *input, void *output);
void AdaptiveAvgPool2d_cpu(NNModule *layer, void *input, void *output);
void transpose_cpu(NNModule *layer, void *input, void *output);
void save_cpu(NNModule *layer, void *input, void *output);
void add_cpu(NNModule *layer, void *input, void *output);

#endif /* _NN_OPS_CPU_H_ */
