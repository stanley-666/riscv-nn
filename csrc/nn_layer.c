#include "nn_layer.h"
#include "nn_utils.h"

static int16_t *pack_linear_weights_i8(const int8_t *wt_src, int inDim, int outDim)
{
    int16_t *wt_rvv = (int16_t *)safe_malloc(inDim * outDim * sizeof(int16_t));
    for (int o = 0; o < outDim; ++o)
        for (int i = 0; i < inDim; ++i)
            wt_rvv[i * outDim + o] = wt_src[o * inDim + i];
    return wt_rvv;
}

static float *pack_linear_weights_f32(const float *wt_src, int inDim, int outDim)
{
    float *wt_rvv = (float *)safe_malloc(inDim * outDim * sizeof(float));
    for (int o = 0; o < outDim; ++o)
        for (int i = 0; i < inDim; ++i)
            wt_rvv[i * outDim + o] = wt_src[o * inDim + i];
    return wt_rvv;
}

void addLayer(CNN *net, NNModule *layer) {
    printf("Adding layer %s (type=%d) to CNN\n", layer_type_name(layer->type), layer->type);
    layer->prev = net->lastModule;
    if (net->lastModule) net->lastModule->next = layer;
    else net->firstModule = layer;
    net->lastModule = layer;
    net->numModules++;
}

NNModule *nn_Conv1d(int inputChannels, int inputLength, int filterSize, int numFilters, int stride, int padding, 
                        ActivationType activation, const void *weights, const void *bias, const void *M, const void *zps, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = CONV1D;
    layer->activation = activation;

    // Shape
    layer->inputShape.N = 1;  // batch size
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = 1;  // height (1D CNN)
    layer->inputShape.W = inputLength;

    layer->outputShape.N = 1;  // batch size
    layer->outputShape.C = numFilters;
    layer->outputShape.H = 1;  // height (1D CNN)
    layer->outputShape.W = (inputLength + 2 * padding - filterSize) / stride + 1; // output length
    // Conv params
    layer->params.conv.filterSize   = filterSize; // kernel size
    layer->params.conv.numFilters   = numFilters; // output channels
    layer->params.conv.stride       = stride;
    layer->params.conv.padding      = padding;

    // Memory
    layer->params.conv.weights = (void *)weights;
    layer->params.conv.weights_rvv = NULL;
    layer->params.conv.bias    = (void *)bias;
    layer->params.conv.M       = (void *)M;
    layer->params.conv.zps     = (void *)zps;
    layer->params.conv.acc_buffer = NULL;

    // rearrange weights for RVV
    layer->dtype = dtype;
    if (dtype == ELEM_INT8) { // pack im2col = (KernelSize * InputChannels, NumFilters)
        int cols = inputChannels * filterSize;
        int outC = numFilters;
        const int8_t *wt_src = (const int8_t *)weights;
        int16_t *wt_rvv = (int16_t *)safe_malloc(outC * cols * sizeof(int16_t));
        for (int oc = 0; oc < outC; ++oc) {
            for (int k = 0; k < filterSize; ++k) {
                for (int ic = 0; ic < inputChannels; ++ic) {
                    int src_idx = oc * inputChannels * filterSize + ic * filterSize + k;
                    int col_idx = k * inputChannels + ic;
                    int dst_idx = col_idx * outC + oc;
                    wt_rvv[dst_idx] = (int16_t)wt_src[src_idx];
                }
            }
        }
        layer->params.conv.weights_rvv = wt_rvv;
        layer->params.conv.acc_buffer = safe_malloc(outC * sizeof(int32_t));
        memset(layer->params.conv.acc_buffer, 0, outC * sizeof(int32_t));
    } else if (dtype == ELEM_FLOAT32) {
        int cols = inputChannels * filterSize;
        int outC = numFilters;
        const float *wt_src = (const float *)weights;
        float *wt_rvv = (float *)safe_malloc(outC * cols * sizeof(float));
        for (int oc = 0; oc < outC; ++oc) {
            for (int k = 0; k < filterSize; ++k) {
                for (int ic = 0; ic < inputChannels; ++ic) {
                    int src_idx = oc * inputChannels * filterSize + ic * filterSize + k;
                    int col_idx = k * inputChannels + ic;
                    int dst_idx = col_idx * outC + oc;
                    wt_rvv[dst_idx] = wt_src[src_idx];
                }
            }
        }
        layer->params.conv.weights_rvv = wt_rvv;
        size_t acc_size = outC * sizeof(float);
        layer->params.conv.acc_buffer = safe_malloc(acc_size);
        memset(layer->params.conv.acc_buffer, 0, acc_size);
    }
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Conv1D NNModule ===\n");
    printf(" Input  : Length=%d, Channels=%d\n", inputLength, inputChannels);
    printf(" Output : Length=%d, Channels=%d\n", layer->outputShape.W, layer->outputShape.C);
    printf(" FilterSize: %d, NumFilters: %d, Stride: %d, Padding: %d\n",
           filterSize, numFilters, stride, padding);
    printf("=========================\n");

    return layer;
}

NNModule *nn_Pool1d(int inputChannels, int inputLength, int poolSize, int stride, PoolType poolType, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = POOL1D;
    layer->activation = NONE;

    // Pool layer params
    layer->params.pool.poolSize = poolSize;
    layer->params.pool.stride   = stride;
    layer->params.pool.type     = poolType;

    // shape
    layer->inputShape.N = 1;  // batch size
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = 1;  // height (1D CNN)
    layer->inputShape.W = inputLength;

    layer->outputShape.N = 1;  // batch size
    layer->outputShape.C = inputChannels;
    layer->outputShape.H = 1;  // height (1D CNN)
    if (poolType == AdaptiveMaxPool1d) {
        layer->outputShape.W = poolSize; // AdaptiveMaxPool1d 輸出長度為 poolSize
    } else {
        layer->outputShape.W = (inputLength - poolSize) / stride + 1;
    }

    layer->dtype = dtype;
    // 不 allocate output，讓 forward 或 ping-pong buffer 處理
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Pool1D NNModule ===\n");
    printf(" Input  : Length=%d, Channels=%d\n", inputLength, inputChannels);
    printf(" Output : Length=%d, Channels=%d\n", layer->outputShape.W, layer->outputShape.C);
    printf(" PoolSize: %d, Stride: %d, Type: %s\n",
           poolSize, stride, poolType == MAX_POOL ? "Max" : "Avg");
    printf("==========================\n");

    return layer;
}

NNModule *nn_Pool2d(int inputChannels, int inputHeight, int inputWidth, int poolSize, int stride, int padding, PoolType poolType, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = POOL2D;
    layer->activation = NONE;

    layer->params.pool2d.poolSize = poolSize;
    layer->params.pool2d.stride = stride;
    layer->params.pool2d.padding = padding;
    layer->params.pool2d.type = poolType;

    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = inputHeight;
    layer->inputShape.W = inputWidth;

    layer->outputShape.N = 1;
    layer->outputShape.C = inputChannels;
    layer->outputShape.H = (inputHeight + 2 * padding - poolSize) / stride + 1;
    layer->outputShape.W = (inputWidth + 2 * padding - poolSize) / stride + 1;

    layer->dtype = dtype;
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Pool2D NNModule ===\n");
    printf(" Input  : H=%d, W=%d, C=%d\n", inputHeight, inputWidth, inputChannels);
    printf(" Output : H=%d, W=%d, C=%d\n", layer->outputShape.H, layer->outputShape.W, layer->outputShape.C);
    printf(" PoolSize: %d, Stride: %d, Padding: %d, Type: %s\n",
           poolSize, stride, padding, poolType == MAX_POOL ? "Max" : "Avg");
    printf("==========================\n");

    return layer;
}

NNModule *nn_AdaptiveAvgPool2d(int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = POOL2D;
    layer->activation = NONE;

    layer->params.pool2d.poolSize = 0;
    layer->params.pool2d.stride = 0;
    layer->params.pool2d.padding = 0;
    layer->params.pool2d.type = AdaptiveAvgPool2d;

    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = inputHeight;
    layer->inputShape.W = inputWidth;

    layer->outputShape.N = 1;
    layer->outputShape.C = inputChannels;
    layer->outputShape.H = outputHeight;
    layer->outputShape.W = outputWidth;

    layer->dtype = dtype;
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added AdaptiveAvgPool2d NNModule ===\n");
    printf(" Input  : H=%d, W=%d, C=%d\n", inputHeight, inputWidth, inputChannels);
    printf(" Output : H=%d, W=%d, C=%d\n", outputHeight, outputWidth, inputChannels);
    printf("==========================\n");

    return layer;
}

NNModule *nn_AdaptiveMaxPool1d(int inputChannels, int inputLength, int outputsize, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = POOL1D;
    layer->activation = NONE;

    // Pool layer params
    layer->params.pool.poolSize = 0; // unused
    layer->params.pool.stride   = 0; // unused
    layer->params.pool.type     = AdaptiveMaxPool1d;

    // shape
    layer->inputShape.N = 1;  // batch size
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = 1;  // height (1D CNN)
    layer->inputShape.W = inputLength;

    layer->outputShape.N = 1;  // batch size
    layer->outputShape.C = inputChannels;
    layer->outputShape.H = 1;  // height (1D CNN)
    layer->outputShape.W = outputsize; // AdaptiveMaxPool1d 輸出長度為 poolSize

    layer->dtype = dtype;
    // 不 allocate output，讓 forward 或 ping-pong buffer 處理
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added AdaptiveMaxPool1d NNModule ===\n");
    printf(" Input  : Length=%d, Channels=%d\n", inputLength, inputChannels);
    printf(" Output : Length=%d, Channels=%d\n", layer->outputShape.W, layer->outputShape.C);
    printf("==========================\n");

    return layer;
}

NNModule *nn_Linear(int inputSize, int outputSize, ActivationType activation, const void *weights, const void *bias, const void *M, const void *zps, elem_type dtype)
{
    if (inputSize <= 0 || outputSize <= 0) {
        printf("Error: Invalid FC dimensions: input=%d, output=%d\n",
               inputSize, outputSize);
        exit(EXIT_FAILURE);
    }

    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = FC;
    layer->activation = activation;

    // Shape
    layer->inputShape.N = 1;  // batch size
    layer->inputShape.C = 1;  // channels
    layer->inputShape.H = 1;  // height (1D CNN)
    layer->inputShape.W = inputSize;

    layer->outputShape.N = 1;  // batch size
    layer->outputShape.C = 1;  // channels
    layer->outputShape.H = 1;  // height (1D CNN)
    layer->outputShape.W = outputSize;

    // FC params
    layer->params.fc.weights    = (void *)weights;
    layer->params.fc.weights_rvv = NULL;
    layer->params.fc.bias       = (void *)bias;
    layer->params.fc.M          = (void *)M;
    layer->params.fc.zps        = (void *)zps;
    layer->params.fc.acc_buffer = NULL;

    layer->dtype = dtype;

    // 預先為 RVV 重排權重，避免推論期 malloc/轉置
    if (dtype == ELEM_INT8) {
        int inDim = inputSize;
        int outDim = outputSize;
        const int8_t *wt_src = (const int8_t *)weights;
        int16_t *wt_rvv = pack_linear_weights_i8(wt_src, inDim, outDim);
        layer->params.fc.weights_rvv = wt_rvv;
        layer->params.fc.acc_buffer = safe_malloc(outDim * sizeof(int32_t));
    } else if (dtype == ELEM_FLOAT32) {
        int inDim = inputSize;
        int outDim = outputSize;
        const float *wt_src = (const float *)weights;
        float *wt_rvv = pack_linear_weights_f32(wt_src, inDim, outDim);
        layer->params.fc.weights_rvv = wt_rvv;
        // float 路徑直接寫入 output，不強制配置 acc_buffer
    }

    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added FC NNModule ===\n");
    printf(" Input : %d, Output: %d\n", layer->inputShape.W, layer->outputShape.W);
    printf("======================\n");

    return layer;
}

NNModule *nn_LayerNorm1d(int inputWidth, int inputChannels, const void *weight, const void *bias, float eps, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = LAYERNORM1D;
    layer->activation = NONE;
    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = 1;
    layer->inputShape.W = inputWidth;
    layer->outputShape = layer->inputShape;
    layer->params.layernorm.weight = (void *)weight;
    layer->params.layernorm.bias = (void *)bias;
    layer->params.layernorm.eps = eps;
    layer->dtype = dtype;
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added LayerNorm1D NNModule ===\n");
    printf(" Input/Output : Width=%d, Channels=%d, eps=%g\n", inputWidth, inputChannels, (double)eps);
    printf("=============================\n");
    return layer;
}

NNModule *nn_MultiHeadAttention1d(int inputWidth,
                                  int embedDim,
                                  int numHeads,
                                  const void *inProjWeight,
                                  const void *inProjBias,
                                  const void *outProjWeight,
                                  const void *outProjBias,
                                  elem_type dtype)
{
    if (dtype != ELEM_FLOAT32 && dtype != ELEM_INT8) {
        printf("Error: unsupported dtype for attention layer\n");
        exit(EXIT_FAILURE);
    }
    if (embedDim <= 0 || numHeads <= 0 || (embedDim % numHeads) != 0) {
        printf("Error: invalid attention dimensions embedDim=%d numHeads=%d\n", embedDim, numHeads);
        exit(EXIT_FAILURE);
    }

    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = ATTENTION1D;
    layer->activation = NONE;
    layer->inputShape.N = 1;
    layer->inputShape.C = embedDim;
    layer->inputShape.H = 1;
    layer->inputShape.W = inputWidth;
    layer->outputShape = layer->inputShape;
    layer->dtype = dtype;

    int qkvDim = embedDim * 3;
    layer->params.attention.in_proj_weight = (void *)inProjWeight;
    layer->params.attention.in_proj_bias = (void *)inProjBias;
    layer->params.attention.out_proj_weight = (void *)outProjWeight;
    layer->params.attention.out_proj_bias = (void *)outProjBias;
    layer->params.attention.num_heads = numHeads;
    layer->params.attention.head_dim = embedDim / numHeads;
    layer->params.attention.scale = 1.0f / sqrtf((float)(embedDim / numHeads));

    if (dtype == ELEM_FLOAT32) {
        layer->params.attention.in_proj_weight_rvv = pack_linear_weights_f32((const float *)inProjWeight, embedDim, qkvDim);
        layer->params.attention.out_proj_weight_rvv = pack_linear_weights_f32((const float *)outProjWeight, embedDim, embedDim);
        layer->params.attention.qkv_buffer = safe_malloc((size_t)inputWidth * qkvDim * sizeof(float));
        layer->params.attention.ctx_buffer = safe_malloc((size_t)inputWidth * embedDim * sizeof(float));
        layer->params.attention.proj_buffer = safe_malloc((size_t)inputWidth * embedDim * sizeof(float));
        layer->params.attention.score_buffer = safe_malloc((size_t)inputWidth * sizeof(float));
    } else {
        layer->params.attention.in_proj_weight_rvv = pack_linear_weights_i8((const int8_t *)inProjWeight, embedDim, qkvDim);
        layer->params.attention.out_proj_weight_rvv = pack_linear_weights_i8((const int8_t *)outProjWeight, embedDim, embedDim);
        layer->params.attention.qkv_buffer = safe_malloc((size_t)inputWidth * qkvDim * sizeof(int8_t));
        layer->params.attention.ctx_buffer = safe_malloc((size_t)inputWidth * embedDim * sizeof(int8_t));
        layer->params.attention.proj_buffer = safe_malloc((size_t)inputWidth * embedDim * sizeof(int8_t));
        layer->params.attention.score_buffer = safe_malloc((size_t)inputWidth * sizeof(float));
    }

    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Attention1D NNModule ===\n");
    printf(" Input/Output : Width=%d, Embed=%d, Heads=%d\n", inputWidth, embedDim, numHeads);
    printf("============================\n");
    return layer;
}

NNModule *nn_Transpose(int width, int channels, TransposeType mode, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = TRANSPOSE;
    layer->activation = NONE;

    layer->inputShape.N = 1;
    layer->inputShape.C = channels;
    layer->inputShape.H = 1;
    layer->inputShape.W = width;

    layer->outputShape = layer->inputShape;
    layer->params.transpose.mode = mode;
    layer->dtype = dtype;
    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Transpose NNModule ===\n");
    printf(" Mode: %s, Width=%d, Channels=%d\n",
           mode == TRANSPOSE_CW_TO_WC ? "CW->WC" : "WC->CW", width, channels);
    printf("======================\n");
    return layer;
}

NNModule *nn_Save(int inputChannels, int inputHeight, int inputWidth, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = RES_SAVE;
    layer->activation = NONE;

    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = inputHeight;
    layer->inputShape.W = inputWidth;
    layer->outputShape = layer->inputShape;
    layer->dtype = dtype;

    layer->params.save.bytes = (size_t)inputChannels * inputHeight * inputWidth * sizeof_dtype(dtype);
    printf("Save layer will save %zu bytes\n", layer->params.save.bytes);
    layer->params.save.buffer = NULL; // shared buffer to be allocated during inference

    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Save NNModule ===\n");
    printf(" Shape : H=%d, W=%d, C=%d\n", inputHeight, inputWidth, inputChannels);
    printf("======================\n");
    return layer;
}

NNModule *nn_Add(int inputChannels, int inputHeight, int inputWidth, ActivationType activation, void *skip, elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = RES_ADD;
    layer->activation = activation;

    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = inputHeight;
    layer->inputShape.W = inputWidth;
    layer->outputShape = layer->inputShape;
    layer->dtype = dtype;

    layer->params.add.bytes = (size_t)inputChannels * inputHeight * inputWidth * sizeof_dtype(dtype);
    layer->params.add.skip = skip;

    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Add NNModule ===\n");
    printf(" Shape : H=%d, W=%d, C=%d\n", inputHeight, inputWidth, inputChannels);
    printf("======================\n");
    return layer;
}

// add conv2d layer

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
                    elem_type dtype)
{
    NNModule *layer = (NNModule *)safe_malloc(sizeof(NNModule));
    layer->type = CONV2D;
    layer->activation = activation;

    layer->inputShape.N = 1;
    layer->inputShape.C = inputChannels;
    layer->inputShape.H = inputHeight;
    layer->inputShape.W = inputWidth;

    layer->outputShape.N = 1;
    layer->outputShape.C = numFilters;
    layer->outputShape.H = (inputHeight + 2 * padding - filterSize) / stride + 1;
    layer->outputShape.W = (inputWidth + 2 * padding - filterSize) / stride + 1;

    layer->params.conv2d.filterSize = filterSize;
    layer->params.conv2d.numFilters = numFilters;
    layer->params.conv2d.stride = stride;
    layer->params.conv2d.padding = padding;

    layer->params.conv2d.weights = (void *)weights;
    layer->params.conv2d.weights_rvv = NULL;
    layer->params.conv2d.bias = (void *)bias;
    layer->params.conv2d.M = (void *)M;
    layer->params.conv2d.zps = (void *)zps;
    layer->params.conv2d.input_override = NULL;
    layer->params.conv2d.acc_buffer = NULL;

    layer->dtype = dtype;
    // pack weights for riscv vector
    if (dtype == ELEM_INT8) { // pack im2col = (KernelSize * KernelSize * InputChannels, NumFilters)
        int cols = inputChannels * filterSize * filterSize;
        int outC = numFilters;
        const int8_t *wt_src = (const int8_t *)weights;
        int16_t *wt_rvv = (int16_t *)safe_malloc(outC * cols * sizeof(int16_t));
        for (int oc = 0; oc < outC; ++oc) {
            for (int kh = 0; kh < filterSize; ++kh) {
                for (int kw = 0; kw < filterSize; ++kw) {
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        int src_idx = (((oc * inputChannels + ic) * filterSize + kh) * filterSize + kw);
                        int col_idx = ((kh * filterSize + kw) * inputChannels + ic);
                        int dst_idx = col_idx * outC + oc;
                        wt_rvv[dst_idx] = (int16_t)wt_src[src_idx];
                    }
                }
            }
        }
        layer->params.conv2d.weights_rvv = wt_rvv;
        layer->params.conv2d.acc_buffer = safe_malloc(outC * sizeof(int32_t));
        memset(layer->params.conv2d.acc_buffer, 0, outC * sizeof(int32_t));
    } else if (dtype == ELEM_FLOAT32) {
        int cols = inputChannels * filterSize * filterSize;
        int outC = numFilters;
        const float *wt_src = (const float *)weights;
        float *wt_rvv = (float *)safe_malloc(outC * cols * sizeof(float));
        for (int oc = 0; oc < outC; ++oc) {
            for (int kh = 0; kh < filterSize; ++kh) {
                for (int kw = 0; kw < filterSize; ++kw) {
                    for (int ic = 0; ic < inputChannels; ++ic) {
                        int src_idx = (((oc * inputChannels + ic) * filterSize + kh) * filterSize + kw);
                        int col_idx = ((kh * filterSize + kw) * inputChannels + ic);
                        int dst_idx = col_idx * outC + oc;
                        wt_rvv[dst_idx] = wt_src[src_idx];
                    }
                }
            }
        }
        layer->params.conv2d.weights_rvv = wt_rvv;
        layer->params.conv2d.acc_buffer = safe_malloc(outC * sizeof(float));
        memset(layer->params.conv2d.acc_buffer, 0, outC * sizeof(float));
    }

    layer->prev = NULL;
    layer->next = NULL;

    printf("=== Added Conv2D NNModule ===\n");
    printf(" Input  : Height=%d, Width=%d, Channels=%d\n", inputHeight, inputWidth, inputChannels);
    printf(" Output : Height=%d, Width=%d, Channels=%d\n",
           layer->outputShape.H, layer->outputShape.W, layer->outputShape.C);
    printf(" FilterSize: %d, NumFilters: %d, Stride: %d, Padding: %d\n",
           filterSize, numFilters, stride, padding);
    printf("=========================\n");

    return layer;
}

/* Forward helpers are implemented in csrc/1dcnn_forward.c */

// 創建新的CNN and initialize it
CNN *createCNN()
{
    CNN *net = (CNN *)safe_malloc(sizeof(CNN));
    if (!net) {
        printf("Failed to allocate memory for CNN\n");
        exit(EXIT_FAILURE);
    }else {
        printf("Successfully allocated memory for CNN model\n");
    }
    net->firstModule = NULL;
    net->lastModule = NULL;
    net->numModules = 0;
    return net;
}

void freeCNN(CNN *net)
{
    NNModule *currentLayer = net->firstModule;

    while (currentLayer != NULL)
    {
        NNModule *nextLayer = currentLayer->next;
        switch (currentLayer->type) {
        case CONV1D:
            safe_free(currentLayer->params.conv.weights_rvv);
            safe_free(currentLayer->params.conv.acc_buffer);
            break;
        case CONV2D:
            safe_free(currentLayer->params.conv2d.weights_rvv);
            safe_free(currentLayer->params.conv2d.acc_buffer);
            break;
        case FC:
            safe_free(currentLayer->params.fc.weights_rvv);
            safe_free(currentLayer->params.fc.acc_buffer);
            break;
        case ATTENTION1D:
            safe_free(currentLayer->params.attention.in_proj_weight_rvv);
            safe_free(currentLayer->params.attention.out_proj_weight_rvv);
            safe_free(currentLayer->params.attention.qkv_buffer);
            safe_free(currentLayer->params.attention.ctx_buffer);
            safe_free(currentLayer->params.attention.proj_buffer);
            safe_free(currentLayer->params.attention.score_buffer);
            break;
        default:
            break;
        }
        safe_free(currentLayer);
	    currentLayer = nextLayer;
    }

    safe_free(net);
}

void printCNN(CNN *net)
{
    printf("\n=== CNN Architecture: ===\n"); // cnn架構
    NNModule *currentLayer = net->firstModule;
    int layerNum = 1;

    while (currentLayer != NULL)
    {
        printf("NNModule %d: ", layerNum++);

        switch (currentLayer->type)
        {
        case CONV1D:
            printf("Convolution NNModule\n");
            printf("  filterSize: %d\n", currentLayer->params.conv.filterSize);
            printf("  numFilters: %d\n", currentLayer->params.conv.numFilters);
            printf("  stride: %d\n", currentLayer->params.conv.stride);
            printf("  padding: %d\n", currentLayer->params.conv.padding);
            break;
        case CONV2D:
            printf("Convolution2D NNModule\n");
            printf("  filterSize: %d\n", currentLayer->params.conv2d.filterSize);
            printf("  numFilters: %d\n", currentLayer->params.conv2d.numFilters);
            printf("  stride: %d\n", currentLayer->params.conv2d.stride);
            printf("  padding: %d\n", currentLayer->params.conv2d.padding);
            break;
        case POOL1D:
            printf("%s Pooling NNModule\n",
                   currentLayer->params.pool.type == MAX_POOL ? "Max" : "Average");
            printf("  poolSize: %d\n", currentLayer->params.pool.poolSize);
            printf("  stride: %d\n", currentLayer->params.pool.stride);
            break;
        case POOL2D:
            printf("%s Pooling2D NNModule\n",
                   currentLayer->params.pool2d.type == MAX_POOL ? "Max" : "Average");
            printf("  poolSize: %d\n", currentLayer->params.pool2d.poolSize);
            printf("  stride: %d\n", currentLayer->params.pool2d.stride);
            printf("  padding: %d\n", currentLayer->params.pool2d.padding);
            break;
        case FC:
            printf("Fully Connected NNModule\n");
            break;
        case LAYERNORM1D:
            printf("LayerNorm1D NNModule\n");
            printf("  width: %d, channels: %d, eps=%g\n",
                   currentLayer->inputShape.W,
                   currentLayer->inputShape.C,
                   (double)currentLayer->params.layernorm.eps);
            break;
        case ATTENTION1D:
            printf("Attention1D NNModule\n");
            printf("  width: %d, embedDim: %d, heads: %d\n",
                   currentLayer->inputShape.W,
                   currentLayer->inputShape.C,
                   currentLayer->params.attention.num_heads);
            break;
        case TRANSPOSE:
            printf("Transpose NNModule\n");
            break;
        case RES_SAVE:
            printf("RES_Save NNModule\n");
            break;
        case RES_ADD:
            printf("RES_Add NNModule\n");
            break;
        }

        printf("  Activation function: ");
        switch (currentLayer->activation)
        {
        case RELU:
            printf("ReLU\n");
            break;
        case SIGMOID:
            printf("Sigmoid\n");
            break;
        case TANH:
            printf("Tanh\n");
            break;
        case LEAKY_RELU:
            printf("Leaky ReLU\n");
            break;
        case SOFTMAX:
            printf("Softmax\n");
            break;
        case GELU:
            printf("GELU\n");
            break;
        case NONE:
            printf("None\n");
            break;
        }

        printf("  Input Shape : N=%d, C=%d, H=%d, W=%d\n",
               currentLayer->inputShape.N,
               currentLayer->inputShape.C,
               currentLayer->inputShape.H,
               currentLayer->inputShape.W);
        printf("  Output Shape: N=%d, C=%d, H=%d, W=%d\n",
               currentLayer->outputShape.N,
               currentLayer->outputShape.C,
               currentLayer->outputShape.H,
               currentLayer->outputShape.W);
        currentLayer = currentLayer->next;
    }
    printf("=========================\n");
}
