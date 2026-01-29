#include "nn_infer_vpu.h"
#include <string.h>
#include <time.h>
#include <stdio.h>

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
                AdaptiveMaxPool1d_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool.type == MAX_POOL) {
                maxpool1d_vpu(currentLayer, src, dst);
            } else {
                avgpool1d_vpu(currentLayer, src, dst);
            }
            break;
        case POOL2D:
            if (currentLayer->params.pool2d.type == AdaptiveAvgPool2d) {
                AdaptiveAvgPool2d_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool2d.type == MAX_POOL) {
                maxpool2d_vpu(currentLayer, src, dst);
            } else {
                avgpool2d_vpu(currentLayer, src, dst);
            }
            break;
        case TRANSPOSE:
            transpose_vpu(currentLayer, src, dst);
            break;
        case FC:
            fullyconnected_vpu(currentLayer, src, dst);
            break;
        case RES_SAVE:
            save_vpu(currentLayer, src, dst);
            break;
        case RES_ADD:
            add_vpu(currentLayer, src, dst);
            break;
        }

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
        //clock_t t0 = clock();
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
                AdaptiveMaxPool1d_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool.type == MAX_POOL) {
                maxpool1d_vpu(currentLayer, src, dst);
            } else {
                avgpool1d_vpu(currentLayer, src, dst);
            }
            break;
        case POOL2D:
            if (currentLayer->params.pool2d.type == AdaptiveAvgPool2d) {
                AdaptiveAvgPool2d_vpu(currentLayer, src, dst);
            } else if (currentLayer->params.pool2d.type == MAX_POOL) {
                maxpool2d_vpu(currentLayer, src, dst);
            } else {
                avgpool2d_vpu(currentLayer, src, dst);
            }
            break;
        case TRANSPOSE:
            transpose_vpu(currentLayer, src, dst);
            break;
        case FC:
            fullyconnected_vpu(currentLayer, src, dst);
            break;
        case RES_SAVE:
            save_vpu(currentLayer, src, dst);
            skip_swap = 1; // keep src for subsequent layers; RES_SAVE is identity
            break;
        case RES_ADD:
            add_vpu(currentLayer, src, dst);
            break;
        }

        //clock_t t1 = clock();
        //double ms = ts_diff_ms(t0, t1);
        //printf("[rvv_f32] layer %d type=%s time=%.3f ms\n", layer_idx, layer_type_name(currentLayer->type), ms);
        layer_idx++;

        if (!skip_swap) {
            void *tmp = src;
            src = dst;
            dst = tmp;
        }
        currentLayer = currentLayer->next;
    }
}
