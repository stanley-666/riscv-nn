/* SPDX-FileContributor: Person: Stanley Lee */
/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
*/

#include <stdint.h>

float fmaxf(float a, float b) {
    return (a >= b) ? a : b;
}

float roundf(float x) {
    if (x >= 0.0f)
        return (float)((int32_t)(x + 0.5f));
    return (float)((int32_t)(x - 0.5f));
}

static float pow2_int(int k) {
    float result = 1.0f;
    if (k >= 0) {
        while (k--)
            result *= 2.0f;
    } else {
        while (k++)
            result *= 0.5f;
    }
    return result;
}

float expf(float x) {
    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;
    if (x > 88.0f)
        return 1.0e38f;
    if (x < -88.0f)
        return 0.0f;

    int k = (int)(x * inv_ln2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * ln2;
    float r2 = r * r;
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    float poly = 1.0f + r + r2 * 0.5f + r3 * (1.0f / 6.0f)
               + r4 * (1.0f / 24.0f) + r5 * (1.0f / 120.0f);

    return poly * pow2_int(k);
}

float tanhf(float x) {
    float e2x = expf(2.0f * x);
    float denom = e2x + 1.0f;
    if (denom == 0.0f)
        return 1.0f;
    return (e2x - 1.0f) / denom;
}
