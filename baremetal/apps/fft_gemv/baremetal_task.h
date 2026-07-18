#ifndef FFT_GEMV_BAREMETAL_TASK_H
#define FFT_GEMV_BAREMETAL_TASK_H

#include <stddef.h>

typedef void (*task_scalar_entry_t)(void);
typedef void (*task_rvv_entry_t)(void);

typedef struct {
    const char *name;
    task_scalar_entry_t run_scalar_single;
    task_scalar_entry_t run_scalar_dataset;
    task_rvv_entry_t run_rvv_single;
    task_rvv_entry_t run_rvv_dataset;
} BaremetalTask;

size_t baremetal_task_count(void);
const BaremetalTask *baremetal_task_get(size_t index);
const BaremetalTask *baremetal_task_default(void);

#endif /* FFT_GEMV_BAREMETAL_TASK_H */
