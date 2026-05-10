#include "nn_param.h"

/* Single definition of global buffers used across the project. */
void *buffer1 = NULL;
void *buffer2 = NULL;


const char *layer_type_name(LayerType type) {
    switch (type) {
    case CONV1D: return "Conv1D";
    case CONV2D: return "Conv2D";
    case POOL1D: return "Pool1D";
    case POOL2D: return "Pool2D";
    case FC:     return "FullyConnected";
    case TRANSPOSE: return "Transpose";
    case RES_SAVE: return "RES_Save";
    case RES_ADD: return "RES_Add";
    case LAYERNORM1D: return "LayerNorm1D";
    case ATTENTION1D: return "Attention1D";
    default:     return "Unknown";
    }
}

size_t forward_input_bytes = 0;
