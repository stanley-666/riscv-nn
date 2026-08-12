/* SPDX-FileContributor: Person: Stanley Lee */
/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

extern void *_sbrk(ptrdiff_t incr);

/* RVV buffers returned by this allocator are aligned to a 64-byte boundary. */
#define MINILIB_ALIGNMENT 64UL

typedef struct BlockHeader {
    size_t size;
    struct BlockHeader *next;
    struct BlockHeader *prev;
    bool free;
} BlockHeader;

#define HEADER_SIZE \
    ((sizeof(BlockHeader) + (MINILIB_ALIGNMENT - 1)) & ~(MINILIB_ALIGNMENT - 1))

static BlockHeader *heap_head;

static inline size_t align_up(size_t value) {
    return (value + (MINILIB_ALIGNMENT - 1)) & ~(MINILIB_ALIGNMENT - 1);
}

static void merge_blocks(BlockHeader *a, BlockHeader *b) {
    a->size += HEADER_SIZE + b->size;
    a->next = b->next;
    if (b->next)
        b->next->prev = a;
}

static BlockHeader *extend_heap(BlockHeader *last, size_t size) {
    size_t total = HEADER_SIZE + size;
    uintptr_t current = (uintptr_t)_sbrk(0);
    uintptr_t aligned = align_up(current);
    ptrdiff_t adjust = (ptrdiff_t)(aligned - current);
    if (adjust && _sbrk(adjust) == (void *)-1)
        return NULL;

    BlockHeader *block = (BlockHeader *)_sbrk((ptrdiff_t)total);
    if (block == (void *)-1)
        return NULL;
    block->size = size;
    block->next = NULL;
    block->prev = last;
    block->free = false;
    if (last)
        last->next = block;
    return block;
}

static void split_block(BlockHeader *block, size_t size) {
    size_t remaining = block->size - size;
    if (remaining <= HEADER_SIZE + MINILIB_ALIGNMENT)
        return;

    uint8_t *base = (uint8_t *)block;
    BlockHeader *next = (BlockHeader *)(base + HEADER_SIZE + size);
    next->size = remaining - HEADER_SIZE;
    next->free = true;
    next->next = block->next;
    next->prev = block;
    if (next->next)
        next->next->prev = next;
    block->next = next;
    block->size = size;
}

static BlockHeader *find_block(size_t size) {
    BlockHeader *current = heap_head;
    BlockHeader *last = NULL;
    while (current) {
        if (current->free && current->size >= size) {
            current->free = false;
            split_block(current, size);
            return current;
        }
        last = current;
        current = current->next;
    }

    BlockHeader *block = extend_heap(last, size);
    if (!heap_head)
        heap_head = block;
    return block;
}

void *malloc(size_t size) {
    if (size == 0)
        return NULL;
    size = align_up(size);
    BlockHeader *block = find_block(size);
    if (!block)
        return NULL;
    return (uint8_t *)block + HEADER_SIZE;
}

void free(void *ptr) {
    if (!ptr)
        return;
    BlockHeader *block = (BlockHeader *)((uint8_t *)ptr - HEADER_SIZE);
    block->free = true;
    if (block->next && block->next->free)
        merge_blocks(block, block->next);
    if (block->prev && block->prev->free)
        merge_blocks(block->prev, block);
}

void *calloc(size_t num, size_t size) {
    size_t total = num * size;
    void *ptr = malloc(total);
    if (!ptr)
        return NULL;
    memset(ptr, 0, total);
    return ptr;
}
