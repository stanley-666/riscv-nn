#ifndef _NN_LAYER_H_
#define _NN_LAYER_H_
#include "nn_param.h"
#include <stdio.h>
#include <stdlib.h>

// create layers
NNModule *nn_Conv1d(int inputChannels, int inputLength, int filterSize, int numFilters, int stride, int padding, ActivationType activation, const void *weights, const void *bias, const void *M, const void *zps, elem_type dtype);
NNModule *nn_Conv2d(int inputChannels,
                    int inputHeight,
                    int inputWidth,
                    int filterSize,
                    int numFilters,
                    int stride,
                    int padding,
                    ActivationType activation,
                    const void *weights,
                    const void *bias,
                    const void *M,
                    const void *zps,
                    elem_type dtype);
NNModule *nn_Pool1d(int inputChannels, int inputLength, int poolSize, int stride, PoolType poolType, elem_type dtype);
NNModule *nn_Pool2d(int inputChannels, int inputHeight, int inputWidth, int poolSize, int stride, int padding, PoolType poolType, elem_type dtype);
NNModule *nn_AdaptiveAvgPool2d(int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth, elem_type dtype);
NNModule *nn_Linear(int inputSize, int outputSize, ActivationType activation, const void *weights, const void *bias, const void *M, const void *zps, elem_type dtype);
NNModule *nn_AdaptiveMaxPool1d(int inputChannels, int inputLength, int outputsize, elem_type dtype);
NNModule *nn_Transpose(int width, int channels, TransposeType mode, elem_type dtype);
NNModule *nn_Save(int inputChannels, int inputHeight, int inputWidth, elem_type dtype);
NNModule *nn_Add(int inputChannels, int inputHeight, int inputWidth, ActivationType activation, void *skip, elem_type dtype);

// model create and free
CNN *createCNN();
void freeCNN(CNN *net);
void printCNN(CNN *net);

// connect each layer
void addLayer(CNN *net, NNModule *layer);

#endif
