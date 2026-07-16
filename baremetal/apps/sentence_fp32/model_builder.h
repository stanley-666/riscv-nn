#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "nn_layer.h"

typedef struct {
    double definition_seconds;
    double addition_seconds;
} ModelBuildStats;

CNN *build_sentence_cnn(ModelBuildStats *stats);

#endif
