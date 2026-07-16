#ifndef _NN_INFER_CPU_H_
#define _NN_INFER_CPU_H_

#include "nn_param.h"
#include <stdlib.h>
// main forward logic
void forward(CNN *net, void *input);

#endif
