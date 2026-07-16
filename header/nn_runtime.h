#ifndef _NN_RUNTIME_H_
#define _NN_RUNTIME_H_

#include <stddef.h>

void *safe_malloc(size_t size);
void *safe_calloc(size_t num, size_t size);
void safe_free(void *ptr);

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

#endif
