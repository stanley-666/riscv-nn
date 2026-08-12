/* SPDX-FileContributor: Person: Stanley Lee */
#include "nn_runtime.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define NN_RUNTIME_ALIGNMENT 64

#ifndef NN_RUNTIME_CPU_HZ
#define NN_RUNTIME_CPU_HZ 1000000000UL
#endif

static size_t aligned_size(size_t size)
{
    return (size + NN_RUNTIME_ALIGNMENT - 1) & ~(NN_RUNTIME_ALIGNMENT - 1);
}

void *safe_malloc(size_t size)
{
    size_t allocation_size = aligned_size(size);
    void *ptr = aligned_alloc(NN_RUNTIME_ALIGNMENT, allocation_size);
    if (!ptr) {
        fprintf(stderr, "aligned_alloc failed for size %zu (alignment %d)\n",
                size, NN_RUNTIME_ALIGNMENT);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    if (size != 0 && num > SIZE_MAX / size) {
        fprintf(stderr, "allocation size overflow: %zu * %zu\n", num, size);
        exit(EXIT_FAILURE);
    }
    size_t allocation_size = aligned_size(num * size);
    void *ptr = aligned_alloc(NN_RUNTIME_ALIGNMENT, allocation_size);
    if (!ptr) {
        fprintf(stderr, "aligned_alloc failed for calloc size %zu (alignment %d)\n",
                allocation_size, NN_RUNTIME_ALIGNMENT);
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, allocation_size);
    return ptr;
}

void safe_free(void *ptr)
{
    free(ptr);
}

uint64_t nn_runtime_read_cycles(void)
{
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles) :: "memory");
    return cycles;
}

uint64_t nn_runtime_cycle_frequency_hz(void)
{
    return NN_RUNTIME_CPU_HZ;
}

double nn_runtime_cycles_to_seconds(uint64_t cycles)
{
    return (double)cycles / (double)nn_runtime_cycle_frequency_hz();
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/
