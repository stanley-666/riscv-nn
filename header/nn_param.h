#ifndef _NN_PARAM_H_

#define _NN_PARAM_H_
#include <stdint.h>
#include <stddef.h>   // 這一行讓 NULL 可以使用

/* 1D CNN parameters and structures

Define 1D CNN layer parameters and structures
include layer types, activation functions, and layer definitions

*/
typedef enum { RELU, SIGMOID, TANH, LEAKY_RELU, SOFTMAX, NONE } ActivationType;
typedef enum { CONV1D, CONV2D, POOL1D, POOL2D, FC, TRANSPOSE, RES_SAVE, RES_ADD } LayerType;
typedef enum { MAX_POOL, AVG_POOL, AdaptiveMaxPool1d, AdaptiveAvgPool2d } PoolType;
typedef enum { TRANSPOSE_CW_TO_WC, TRANSPOSE_WC_TO_CW } TransposeType;
/* Adjust element types in layer */
typedef enum { ELEM_INT8, ELEM_INT16, ELEM_FLOAT16, ELEM_FLOAT32} elem_type;
/* 1D CNN layer parameters */
typedef struct { // CONV1D layer parameters
    void *weights;
    void *weights_rvv; // im2col (pack)
    void *bias;
    void *M;           // float* 每通道 scale_in * scale_w / scale_out
    void *zps;        // int32_t* 每通道 zero point
    int sizeofWeights;
    int sizeofBias;
    int filterSize, stride, padding, numFilters;
    void *acc_buffer; // 用於儲存中間累加結果
} ConvParams;

typedef struct { // CONV2D layer parameters
    void *weights;
    void *weights_rvv; // im2col (pack)
    void *bias;
    void *M;           // float* 每通道 scale_in * scale_w / scale_out
    void *zps;        // int32_t* 每通道 zero point
    void *input_override; // optional input pointer for residual branches
    int sizeofWeights;
    int sizeofBias;
    int filterSize, stride, padding, numFilters;
    void *acc_buffer; // 用於儲存中間累加結果
} Conv2DParams;
 
typedef struct {
    int poolSize, stride;
    PoolType type;
} PoolParams;

typedef struct {
    int poolSize, stride, padding;
    PoolType type;
} Pool2DParams;

typedef struct {
    void *weights;
    void *weights_rvv; // transposed weights for RVV
    void *bias;
    void *M;           // float* 每通道 scale_in * scale_w / scale_out
    void *zps;        // int32_t* 每通道 zero point
    int sizeofWeights;
    int sizeofBias;
    void *acc_buffer; // 避免推論期重複分配的累加暫存
} FCParams;

typedef struct {
    TransposeType mode;
} TransposeParams;

typedef struct {
    void *buffer;
    size_t bytes;
} SaveParams;

typedef struct {
    void *skip;
    size_t bytes;
} AddParams;

typedef struct {
    int N;  // batch size
    int C;  // channels
    int H;  // height (1D CNN 1)
    int W;  // width  (1D CNN length)
} TensorShape;

typedef struct NNModule {
    LayerType type;
    ActivationType activation;
    TensorShape inputShape;
    TensorShape outputShape;
    elem_type dtype;
    union {
        ConvParams conv;
        Conv2DParams conv2d;
        PoolParams pool;
        Pool2DParams pool2d;
        FCParams fc;
        TransposeParams transpose;
        // residual branch
        SaveParams save;
        AddParams add;
    } params;
    struct NNModule *prev, *next;
} NNModule;

typedef struct {
    NNModule *firstModule, *lastModule;
    int numModules;
} CNN;

const char *layer_type_name(LayerType type);

/* Buffers used as ping-pong working memory.
    Declared extern here and defined in a single C file to avoid
    multiple definition linker errors. */
extern void *buffer1;
extern void *buffer2;
extern size_t forward_input_bytes;
#endif // _1DCNN_PARAM_H_

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University. All rights reserved.
Author : Stanley Lee
*/
