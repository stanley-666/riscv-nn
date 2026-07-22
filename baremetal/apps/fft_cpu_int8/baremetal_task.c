#include "baremetal_task.h"
#include "fft_cpu_int8_runner.h"

static void run_fft_cpu_int8(void)
{
    (void)fft_cpu_int8_testbench_run();
}

static const BaremetalTask tasks[] = {{
    .name = "fft_cpu_int8",
    .run_scalar_single = run_fft_cpu_int8,
    .run_scalar_dataset = NULL,
    .run_rvv_single = NULL,
    .run_rvv_dataset = NULL,
}};

size_t baremetal_task_count(void) { return sizeof(tasks) / sizeof(tasks[0]); }
const BaremetalTask *baremetal_task_get(size_t index)
{
    return index < baremetal_task_count() ? &tasks[index] : NULL;
}
const BaremetalTask *baremetal_task_default(void) { return baremetal_task_get(0); }
