#include "nn_infer_cpu.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

void forward(CNN *net, void *input)
{
    if (forward_input_bytes > 0)
        memcpy(buffer1, input, forward_input_bytes);
    else 
        return;
    void *src = buffer1;
    void *dst = buffer2;
    clock_t start_time, end_time;
    double elapsed_time;
    NNModule *currentLayer = net->firstModule;
    while (currentLayer != NULL) {
        switch (currentLayer->type) {
        case CONV1D:
            start_time = clock();
            conv1d_cpu(currentLayer, src, dst);
            end_time = clock();
            elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Conv1D Layer Inference time: %.6f seconds\n", elapsed_time);
            break;
        case CONV2D:
            start_time = clock();
            {
                void *conv_in = currentLayer->params.conv2d.input_override ? currentLayer->params.conv2d.input_override : src;
                conv2d_cpu(currentLayer, conv_in, dst);
            }
            end_time = clock();
            elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Conv2D Layer Inference time: %.6f seconds\n", elapsed_time);
            break;
        case POOL1D:
            if (currentLayer->params.pool.type == AdaptiveMaxPool1d)
                AdaptiveMaxPool1d_cpu(currentLayer, src, dst);
            else
                pool1d_cpu(currentLayer, src, dst);
            break;
        case POOL2D:
            if (currentLayer->params.pool2d.type == AdaptiveAvgPool2d)
                AdaptiveAvgPool2d_cpu(currentLayer, src, dst);
            else
                pool2d_cpu(currentLayer, src, dst);
            break;
        case TRANSPOSE:
            start_time = clock();
            transpose_cpu(currentLayer, src, dst);
            end_time = clock();
            elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("Transpose Layer Inference time: %.6f seconds\n", elapsed_time);
            break;
        case FC:
            start_time = clock();
            fullyconnected_cpu(currentLayer, src, dst);
            end_time = clock();
            elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
            printf("FC Layer Inference time: %.6f seconds\n", elapsed_time);
            break;
        case LAYERNORM1D:
            layernorm1d_cpu(currentLayer, src, dst);
            break;
        case ATTENTION1D:
            attention1d_cpu(currentLayer, src, dst);
            break;
        case RES_SAVE:
            save_cpu(currentLayer, src, dst);
            break;
        case RES_ADD:
            add_cpu(currentLayer, src, dst);
            break;
        default:
            printf("Unsupported layer type %d in forward()\n", currentLayer->type);
            exit(EXIT_FAILURE);
        }

        void *tmp = src;
        src = dst;
        dst = tmp;

        currentLayer = currentLayer->next;
    }
}
