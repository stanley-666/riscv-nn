#include "baremetal_task.h"
#include "cnn_demo.h"

#ifndef DEFAULT_TASK_INDEX
#define DEFAULT_TASK_INDEX 0
#endif

static const BaremetalTask tasks[] = {
    {
        .name = "sentence_inference",
        .run_scalar_single = NULL,
        .run_scalar_dataset = NULL,
        .run_rvv_single = run_rvv_single_demo,
        .run_rvv_dataset = NULL,
    },
};

size_t baremetal_task_count(void) {
    return sizeof(tasks) / sizeof(tasks[0]);
}

const BaremetalTask *baremetal_task_get(size_t index) {
    size_t count = baremetal_task_count();
    if (count == 0 || index >= count)
        return NULL;
    return &tasks[index];
}

const BaremetalTask *baremetal_task_default(void) {
    const BaremetalTask *task = baremetal_task_get(DEFAULT_TASK_INDEX);
    if (!task)
        task = baremetal_task_get(0);
    return task;
}
