/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FFT_MIXED_RADIX_Q7_H
#define FFT_MIXED_RADIX_Q7_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define FFT_MIXED_Q7_MAX_RADIX 16
#define FFT_MIXED_Q7_TRANSFORMS 69

typedef struct {
    int8_t coefficient[2][2][FFT_MIXED_Q7_MAX_RADIX]
                              [FFT_MIXED_Q7_MAX_RADIX];
    uint16_t span;
    uint16_t previous_span;
    uint8_t radix;
    uint8_t log2_radix;
    uint8_t offset;
} fft_mixed_q7_transform_t;

typedef struct {
    fft_mixed_q7_transform_t transform[FFT_MIXED_Q7_TRANSFORMS];
    uint16_t count;
} fft_mixed_q7_plan_t;

static inline int8_t fft_mixed_q7_quantize(float value)
{
    long scaled = lroundf(value * 128.0f);
    if (scaled > 127) scaled = 127;
    if (scaled < -128) scaled = -128;
    return (int8_t)scaled;
}

static inline int32_t fft_mixed_q7_rnu_shift(
    int32_t value, unsigned shift)
{
    return (value + (INT32_C(1) << (shift - 1))) >> shift;
}

static inline int8_t fft_mixed_q7_saturate(int32_t value)
{
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static inline size_t fft_mixed_q7_source_index(size_t destination)
{
    size_t digit0 = destination & 3;
    size_t digit1 = (destination >> 2) & 15;
    size_t digit2 = destination >> 6;
    return digit0 * 256 + digit1 * 16 + digit2;
}

static inline void fft_mixed_q7_plan_init(fft_mixed_q7_plan_t *plan)
{
    static const uint8_t radices[] = {4, 16, 16};
    static const uint8_t log2_radices[] = {2, 4, 4};
    size_t next = 0;
    size_t previous_span = 1;
    for (size_t stage = 0; stage < 3; ++stage) {
        size_t radix = radices[stage];
        size_t span = previous_span * radix;
        for (size_t offset = 0; offset < previous_span; ++offset) {
            fft_mixed_q7_transform_t *transform = &plan->transform[next++];
            transform->span = (uint16_t)span;
            transform->previous_span = (uint16_t)previous_span;
            transform->radix = (uint8_t)radix;
            transform->log2_radix = log2_radices[stage];
            transform->offset = (uint8_t)offset;
            for (size_t output = 0; output < radix; ++output)
                for (size_t input = 0; input < radix; ++input) {
                    float angle =
                        -2.0f * (float)M_PI * (float)input *
                        (float)(output * previous_span + offset) /
                        (float)span;
                    int8_t wr = fft_mixed_q7_quantize(cosf(angle));
                    transform->coefficient[0][0][output][input] = wr;
                    transform->coefficient[0][1][output][input] =
                        fft_mixed_q7_quantize(-sinf(angle));
                    transform->coefficient[1][0][output][input] =
                        fft_mixed_q7_quantize(sinf(angle));
                    transform->coefficient[1][1][output][input] = wr;
                }
        }
        previous_span = span;
    }
    plan->count = (uint16_t)next;
}

static inline void fft_mixed_q7_scalar(
    const fft_mixed_q7_plan_t *plan, int8_t *real, int8_t *imag,
    size_t size)
{
    int8_t permuted_real[1024];
    int8_t permuted_imag[1024];
    for (size_t destination = 0; destination < size; ++destination) {
        size_t source = fft_mixed_q7_source_index(destination);
        permuted_real[destination] = real[source];
        permuted_imag[destination] = imag[source];
    }
    for (size_t index = 0; index < size; ++index) {
        real[index] = permuted_real[index];
        imag[index] = permuted_imag[index];
    }

    for (size_t index = 0; index < plan->count; ++index) {
        const fft_mixed_q7_transform_t *transform = &plan->transform[index];
        size_t radix = transform->radix;
        size_t span = transform->span;
        size_t previous_span = transform->previous_span;
        size_t offset = transform->offset;
        for (size_t block = 0; block < size; block += span) {
            int8_t output_real[FFT_MIXED_Q7_MAX_RADIX];
            int8_t output_imag[FFT_MIXED_Q7_MAX_RADIX];
            for (size_t output = 0; output < radix; ++output) {
                int32_t real_acc = 0, imag_acc = 0;
                for (size_t input = 0; input < radix; ++input) {
                    size_t bin = block + offset + input * previous_span;
                    int32_t br = real[bin], bi = imag[bin];
                    real_acc +=
                        transform->coefficient[0][0][output][input] * br;
                    real_acc +=
                        transform->coefficient[0][1][output][input] * bi;
                    imag_acc +=
                        transform->coefficient[1][0][output][input] * br;
                    imag_acc +=
                        transform->coefficient[1][1][output][input] * bi;
                }
                unsigned shift = 7 + transform->log2_radix;
                output_real[output] = fft_mixed_q7_saturate(
                    fft_mixed_q7_rnu_shift(real_acc, shift));
                output_imag[output] = fft_mixed_q7_saturate(
                    fft_mixed_q7_rnu_shift(imag_acc, shift));
            }
            for (size_t output = 0; output < radix; ++output) {
                size_t bin = block + offset + output * previous_span;
                real[bin] = output_real[output];
                imag[bin] = output_imag[output];
            }
        }
    }
}

#endif /* FFT_MIXED_RADIX_Q7_H */
