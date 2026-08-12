/* SPDX-FileContributor: Person: Stanley Lee */
#ifndef _NN_RUNTIME_H_
#define _NN_RUNTIME_H_

#include <stddef.h>
#include <stdint.h>

void *safe_malloc(size_t size);
void *safe_calloc(size_t num, size_t size);
void safe_free(void *ptr);
uint64_t nn_runtime_read_cycles(void);
uint64_t nn_runtime_cycle_frequency_hz(void);
double nn_runtime_cycles_to_seconds(uint64_t cycles);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
