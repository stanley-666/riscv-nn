#ifndef _NN_INFER_CPU_H_
#define _NN_INFER_CPU_H_

#include "nn_ops_cpu.h"
#include <stdlib.h>
// main forward logic
void forward(CNN *net, void *input);

#endif