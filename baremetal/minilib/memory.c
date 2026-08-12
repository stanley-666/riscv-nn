/* SPDX-FileContributor: Person: Stanley Lee */
/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dest, const void *src, size_t len) {
    uintptr_t mask = sizeof(uintptr_t) - 1;
    if ((((uintptr_t)dest | (uintptr_t)src | len) & mask) == 0) {
        const uintptr_t *s = (const uintptr_t *)src;
        uintptr_t *d = (uintptr_t *)dest;
        uintptr_t *end = d + (len / sizeof(uintptr_t));
        while (d < end)
            *d++ = *s++;
    } else {
        const unsigned char *s = (const unsigned char *)src;
        unsigned char *d = (unsigned char *)dest;
        const unsigned char *end = d + len;
        while (d < end)
            *d++ = *s++;
    }
    return dest;
}

void *memset(void *dest, int value, size_t len) {
    uintptr_t mask = sizeof(uintptr_t) - 1;
    if ((((uintptr_t)dest | len) & mask) == 0) {
        uintptr_t word = (unsigned char)value;
        for (size_t shift = 8; shift < sizeof(uintptr_t) * 8; shift <<= 1)
            word |= word << shift;
        uintptr_t *d = (uintptr_t *)dest;
        uintptr_t *end = d + (len / sizeof(uintptr_t));
        while (d < end)
            *d++ = word;
    } else {
        unsigned char *d = (unsigned char *)dest;
        const unsigned char *end = d + len;
        unsigned char byte = (unsigned char)value;
        while (d < end)
            *d++ = byte;
    }
    return dest;
}

int memcmp(const void *s1, const void *s2, size_t n) {
    uintptr_t mask = sizeof(uintptr_t) - 1;
    if ((((uintptr_t)s1 | (uintptr_t)s2) & mask) == 0) {
        const uintptr_t *u1 = (const uintptr_t *)s1;
        const uintptr_t *u2 = (const uintptr_t *)s2;
        size_t words = n / sizeof(uintptr_t);
        while (words--) {
            if (*u1 != *u2) {
                size_t consumed = (const unsigned char *)u1 - (const unsigned char *)s1;
                s1 = (const unsigned char *)u1;
                s2 = (const unsigned char *)u2;
                n -= consumed;
                goto byte_compare;
            }
            ++u1;
            ++u2;
        }
        size_t consumed = (const unsigned char *)u1 - (const unsigned char *)s1;
        s1 = u1;
        s2 = u2;
        n -= consumed;
    }

byte_compare:
    while (n--) {
        unsigned char c1 = *(const unsigned char *)s1++;
        unsigned char c2 = *(const unsigned char *)s2++;
        if (c1 != c2)
            return (int)c1 - (int)c2;
    }
    return 0;
}
