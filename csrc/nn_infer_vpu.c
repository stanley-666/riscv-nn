#include "nn_infer_vpu.h"
#include <string.h>
#include <time.h>
#include <stdio.h>

#define LOGITS_PRINT_COUNT 8

static void dump_layer_logits_i8(const NNModule *layer, int layer_idx, const void *tensor)
{
    /* Skip dump for RES_SAVE because output buffer is intentionally untouched. */
    if (layer->type == RES_SAVE) {
        printf("[int8][layer %d %s] skip dump (no direct output)\n", layer_idx, layer_type_name(layer->type));
        return;
    }

    const int8_t *data = (const int8_t *)tensor;
    size_t total = (size_t)layer->outputShape.N * layer->outputShape.C * layer->outputShape.H * layer->outputShape.W;

    if (!data || total == 0) {
        printf("[int8][layer %d %s] logits unavailable (total=%zu)\n", layer_idx, layer_type_name(layer->type), total);
        return;
    }

    size_t to_print = total < LOGITS_PRINT_COUNT ? total : LOGITS_PRINT_COUNT;
    printf("[int8][layer %d %s] logits first %zu/%zu: ", layer_idx, layer_type_name(layer->type), to_print, total);
    for (size_t i = 0; i < to_print; ++i)
        printf("%d ", data[i]);
    printf("\n");
}

void forward_int8_vpu(CNN *net, void *input)
{
    
    if (forward_input_bytes > 0)
        memcpy(buffer1, input, forward_input_bytes);
    else 
        return;
    NNModule *currentLayer = net->firstModule;
    void *src = buffer1;
    void *dst = buffer2;
    int layer_idx = 0;
    while (currentLayer != NULL)
    {    
        switch (currentLayer->type)
        {
        case CONV1D:        
            conv1d_i8_vpu(currentLayer, src, dst);
            break;
        case CONV2D:
            {
                void *conv_in = src;
                if (currentLayer->params.conv2d.input_override)
                    conv_in = currentLayer->params.conv2d.input_override;
                conv2d_int8_vpu(currentLayer, conv_in, dst);
            }
            break;
        case POOL1D:
            if (currentLayer->params.pool.type == AdaptiveMaxPool1d) {
                AdaptiveMaxPool1d_wc_int8_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool.type == MAX_POOL) {
                maxpool1d_int8_vpu(currentLayer, src, dst);
            } else {
                avgpool1d_int8_vpu(currentLayer, src, dst);
            }
            break;
        case POOL2D:
            if (currentLayer->params.pool2d.type == AdaptiveAvgPool2d) {
                AdaptiveAvgPool2d_int8_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool2d.type == MAX_POOL) {
                maxpool2d_int8_vpu(currentLayer, src, dst);
            } else {
                avgpool2d_int8_vpu(currentLayer, src, dst);
            }
            break;
        case TRANSPOSE:
            transpose_vpu(currentLayer, src, dst);
            break;
        case FC:
            fullyconnected_int8_vpu(currentLayer, src, dst);
            break;
        case LAYERNORM1D:
            printf("LayerNorm1D int8 path is unsupported\n");
            exit(EXIT_FAILURE);
        case ATTENTION1D:
            printf("Attention1D int8 path is unsupported\n");
            exit(EXIT_FAILURE);
        case RES_SAVE:
            save_vpu(currentLayer, src, dst);
            break;
        case RES_ADD:
            add_vpu(currentLayer, src, dst);
            break;
        }

        //dump_layer_logits_i8(currentLayer, layer_idx, dst);

        void *tmp = src;
        src = dst;
        dst = tmp;
        currentLayer = currentLayer->next;
        layer_idx++;
    }
}

void forward_fp32_vpu(CNN *net, void *input)
{
    NNModule *currentLayer = net->firstModule;
    if (forward_input_bytes > 0)
        memcpy(buffer1, input, forward_input_bytes);
    else 
        return;
    void *src = buffer1;
    void *dst = buffer2;
    int layer_idx = 0;
    while (currentLayer != NULL)
    {    
        int skip_swap = 0;
        switch (currentLayer->type)
        {
        case CONV1D:   
            conv1d_fp32_vpu(currentLayer, src, dst);
            break;
        case CONV2D:
            {
                void *conv_in = src;
                if (currentLayer->params.conv2d.input_override)
                    conv_in = currentLayer->params.conv2d.input_override;
                conv2d_fp32_vpu(currentLayer, conv_in, dst);
            }
            break;
        case POOL1D:      
            if (currentLayer->params.pool.type == AdaptiveMaxPool1d) {
                AdaptiveMaxPool1d_wc_fp32_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool.type == MAX_POOL) {
                maxpool1d_fp32_vpu(currentLayer, src, dst);
            } else {
                avgpool1d_fp32_vpu(currentLayer, src, dst);
            }
            break;
        case POOL2D:
            if (currentLayer->params.pool2d.type == AdaptiveAvgPool2d) {
                AdaptiveAvgPool2d_fp32_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool2d.type == MAX_POOL) {
                maxpool2d_fp32_vpu(currentLayer, src, dst);
            } else {
                avgpool2d_fp32_vpu(currentLayer, src, dst);
            }
            break;
        case TRANSPOSE:
            transpose_vpu(currentLayer, src, dst);
            break;
        case FC:
            fullyconnected_fp32_vpu(currentLayer, src, dst);
            break;
        case LAYERNORM1D:
            layernorm1d_fp32_vpu(currentLayer, src, dst);
            break;
        case ATTENTION1D:
            attention1d_fp32_vpu(currentLayer, src, dst);
            break;
        case RES_SAVE:
            save_vpu(currentLayer, src, dst);
            skip_swap = 1; // keep src for subsequent layers; RES_SAVE is identity
            break;
        case RES_ADD:
            add_vpu(currentLayer, src, dst);
            break;
        }
        layer_idx++;

        if (!skip_swap) {
            void *tmp = src;
            src = dst;
            dst = tmp;
        }
        currentLayer = currentLayer->next;
    }
}
