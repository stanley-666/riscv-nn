#include "model_builder.h"
#include "baremetal_timer.h"
#include "sentence_model.h"

CNN *build_sentence_cnn(ModelBuildStats *stats)
{
    timer_ticks_t def_start = timer_now();
    SentenceFp32Layers layers = sentence_fp32_define_layers();
    timer_ticks_t def_end = timer_now();

    CNN *model = sentence_fp32_connect_layers(&layers);
    timer_ticks_t add_end = timer_now();

    if (stats) {
        stats->definition_seconds = timer_to_seconds(def_end - def_start);
        stats->addition_seconds = timer_to_seconds(add_end - def_end);
    }
    return model;
}
