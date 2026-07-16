/*
 * Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "baremetal_task.h"
#include "cnn_demo.h"

static const BaremetalTask tasks[] = {
    {
        .name = "sentence_inference_int8",
        .run_scalar_single = NULL,
        .run_scalar_dataset = NULL,
        .run_rvv_single = run_rvv_single_demo,
        .run_rvv_dataset = NULL,
    },
};

size_t baremetal_task_count(void)
{
    return sizeof(tasks) / sizeof(tasks[0]);
}

const BaremetalTask *baremetal_task_get(size_t index)
{
    return index < baremetal_task_count() ? &tasks[index] : NULL;
}

const BaremetalTask *baremetal_task_default(void)
{
    return baremetal_task_get(0);
}
