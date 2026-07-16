#ifndef BAREMETAL_TIMER_H
#define BAREMETAL_TIMER_H

#include <stdint.h>

#if defined(BAREMETAL) && !defined(BAREMETAL_CPU_HZ)
#define BAREMETAL_CPU_HZ 50000000UL
#endif

typedef uint64_t timer_ticks_t;

static inline timer_ticks_t timer_now(void) {
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline double timer_to_seconds(timer_ticks_t delta) {
    return (double)delta / (double)BAREMETAL_CPU_HZ;
}

#endif
