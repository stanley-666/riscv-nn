#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "baremetal_task.h"
#include "baremetal_timer.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#if !defined(BAREMETAL)
#error "This build is intended for bare-metal targets only."
#endif

#ifndef ENABLE_VECTOR
#define ENABLE_VECTOR 0
#endif

#ifndef BUILD_DATE
#define BUILD_DATE __DATE__
#endif

#ifndef BUILD_TIME
#define BUILD_TIME __TIME__
#endif

static inline uint64_t read_misa(void) {
    uint64_t misa = 0;
    __asm__ volatile("csrr %0, misa" : "=r"(misa));
    return misa;
}

static inline void enable_rvv_context(void) {
#if defined(__riscv_vector)
    uint64_t mstatus;
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    const uint64_t VS_DIRTY = (uint64_t)3 << 9;
    if ((mstatus & VS_DIRTY) != VS_DIRTY) {
        mstatus |= VS_DIRTY;
        __asm__ volatile("csrw mstatus, %0" :: "r"(mstatus));
    }
    __asm__ volatile("vsetvli zero, zero, e8, m1, ta, ma");
#endif
}

static inline bool rvv_extension_present(void) {
    // check misa register for 'V' bit
    uint64_t misa = read_misa();
    const uint64_t V_BIT = 1ULL << (('V' - 'A'));
    return (misa & V_BIT) != 0;
}

static inline void log_vlen_once(void) {
#if defined(__riscv_vector)
    static int printed;
    if (!printed) {
        size_t vlen_bytes = __riscv_vlenb();
        printf("[rvv] VLEN = %lu bytes (%lu bits)\n",
               (unsigned long)vlen_bytes,
               (unsigned long)(vlen_bytes * 8));
        printed = 1;
    }
#else
    (void)0;
#endif
}

static void print_hardware_config(bool rvv_available) {
    uint64_t misa = read_misa();
    printf("Hardware config:\n");
    printf("  misa                 : 0x%016lx\n", (unsigned long)misa);
#if defined(__riscv_xlen)
    printf("  XLEN                 : %d\n", __riscv_xlen);
#endif
#if defined(__riscv_flen)
    printf("  FLEN                 : %d\n", __riscv_flen);
#endif
    printf("  BAREMETAL_CPU_HZ     : %lu\n", (unsigned long)BAREMETAL_CPU_HZ);
#if defined(__riscv_vector)
    printf("  RVV (build flag)     : enabled\n");
#else
    printf("  RVV (build flag)     : disabled\n");
#endif
    printf("  RVV (misa detected)  : %s\n", rvv_available ? "yes" : "no");
}

static void run_scalar_path(const BaremetalTask *task) {
    if (!task)
        return;
    if (task->run_scalar_single)
        task->run_scalar_single();
    if (task->run_scalar_dataset)
        task->run_scalar_dataset();
}

static void run_rvv_path(const BaremetalTask *task) {
    if (!task)
        return;
    bool executed = false;
    if (task->run_rvv_single) {
        task->run_rvv_single();
        executed = true;
    }
    if (task->run_rvv_dataset) {
        task->run_rvv_dataset();
        executed = true;
    }
    if (!executed) {
        printf("RVV entries unavailable for task \"%s\". Falling back to scalar path.\n",
               task->name ? task->name : "unknown");
        run_scalar_path(task);
    }
}

int main(void) {
    printf("Build timestamp: %s %s\n", BUILD_DATE, BUILD_TIME);
    const BaremetalTask *task = baremetal_task_default();
    if (!task) {
        printf("No bare-metal tasks registered.\n");
        return 1;
    }

    bool rvv_available = rvv_extension_present();
    print_hardware_config(rvv_available);

    printf("Selected task: %s\n", task->name ? task->name : "(unnamed)");

    if (rvv_available && ENABLE_VECTOR) {
        enable_rvv_context();
        printf("RVV context enabled.\n");
        log_vlen_once();
        run_rvv_path(task);
    } else {
        if (!rvv_available)
            printf("RVV extension not detected. Running scalar inference.\n");
        else
            printf("RVV support disabled. Running scalar inference.\n");
        run_scalar_path(task);
    }
    return 0;
}
