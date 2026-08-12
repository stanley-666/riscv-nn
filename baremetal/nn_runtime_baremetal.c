/* SPDX-FileContributor: Person: Stanley Lee */
#include "nn_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef BAREMETAL_CPU_HZ
#define BAREMETAL_CPU_HZ 50000000UL
#endif

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr) {
        printf("malloc failed for size %u\n", (unsigned int)size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size)
{
    if (size != 0 && num > SIZE_MAX / size) {
        printf("allocation size overflow\n");
        exit(EXIT_FAILURE);
    }
    size_t allocation_size = num * size;
    void *ptr = malloc(allocation_size);
    if (!ptr) {
        printf("malloc failed for calloc size %u\n", (unsigned int)allocation_size);
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
    return BAREMETAL_CPU_HZ;
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
