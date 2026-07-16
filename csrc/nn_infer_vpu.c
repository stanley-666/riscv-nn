#include "nn_infer_vpu.h"
#include "nn_ops.h"
#include "backends/riscv/vector/ops/conv1d/nn_ops_vpu_conv1d_fp32_internal.h"
#include "backends/riscv/vector/ops/conv1d/nn_ops_vpu_conv1d_i8_internal.h"
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define LOGITS_PRINT_COUNT 8

static inline uint64_t inference_read_cycle(void)
{
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

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
            conv1d_i8_vpu_im2col_unroll8_acc2_m8(currentLayer, src, dst);
            break;
        case CONV2D:
            {
                void *conv_in = src;
                if (currentLayer->params.conv2d.input_override)
                    conv_in = currentLayer->params.conv2d.input_override;
            conv2d_int8_vpu_im2col_m8(currentLayer, conv_in, dst);
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
    forward_fp32_vpu_profile(net, input, NULL);
}

void forward_fp32_vpu_profile(CNN *net, void *input, NNInferenceProfile *profile)
{
    uint64_t total_start = 0;
    NNModule *currentLayer = net->firstModule;
    if (profile) {
        total_start = inference_read_cycle();
        profile->total_cycles = 0;
        profile->num_layers = 0;
    }
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
        uint64_t layer_start = 0;
        if (profile)
            layer_start = inference_read_cycle();
        switch (currentLayer->type)
        {
        case CONV1D:   
            conv1d_fp32_vpu_im2col_unroll8_acc2_m8(currentLayer, src, dst);
            break;
        case CONV2D:
            {
                void *conv_in = src;
                if (currentLayer->params.conv2d.input_override)
                    conv_in = currentLayer->params.conv2d.input_override;
            conv2d_fp32_vpu_im2col(currentLayer, conv_in, dst);
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
        if (profile && layer_idx < NN_INFERENCE_PROFILE_MAX_LAYERS) {
            uint64_t layer_end = inference_read_cycle();
            profile->layers[layer_idx].type = currentLayer->type;
            profile->layers[layer_idx].cycles = layer_end - layer_start;
            profile->num_layers = layer_idx + 1;
        }
        layer_idx++;

        if (!skip_swap) {
            void *tmp = src;
            src = dst;
            dst = tmp;
        }
        currentLayer = currentLayer->next;
    }
    if (profile)
        profile->total_cycles = inference_read_cycle() - total_start;
}
