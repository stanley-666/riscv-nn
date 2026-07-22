/* SPDX-License-Identifier: Apache-2.0 */
/* Scalar radix-2 FFT using the same vectors and tolerances as the RVV test. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "fft_vectors.h"

#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_BATCH_COUNT 64
#define FFT_MAX_MISMATCH_REPORTS 64

static float actual_real[FFT_BATCH_COUNT][FFT_SIZE]
    __attribute__((aligned(64)));
static float actual_imag[FFT_BATCH_COUNT][FFT_SIZE]
    __attribute__((aligned(64)));

static uint64_t read_cycles(void)
{
#if defined(__riscv)
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
#else
    return (uint64_t)clock();
#endif
}

static unsigned reverse_bits(unsigned value, unsigned bits)
{
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

static void fft_cpu_f32(float *real, float *imag, size_t size)
{
    unsigned bits = 0;
    for (size_t value = size; value > 1; value >>= 1) {
        ++bits;
    }

    for (size_t index = 0; index < size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, bits);
        if (reversed > index) {
            float tmp = real[index];
            real[index] = real[reversed];
            real[reversed] = tmp;
            tmp = imag[index];
            imag[index] = imag[reversed];
            imag[reversed] = tmp;
        }
    }

    for (size_t block_size = 2; block_size <= size; block_size <<= 1) {
        size_t half = block_size >> 1;
        for (size_t block = 0; block < size; block += block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                float angle = -2.0f * (float)M_PI * (float)offset / (float)block_size;
                float wr = cosf(angle);
                float wi = sinf(angle);
                size_t even = block + offset;
                size_t odd = even + half;
                float odd_real = real[odd];
                float odd_imag = imag[odd];
                float product_real = wr * odd_real - wi * odd_imag;
                float product_imag = wr * odd_imag + wi * odd_real;
                float even_real = real[even];
                float even_imag = imag[even];

                real[even] = even_real + product_real;
                imag[even] = even_imag + product_imag;
                real[odd] = even_real - product_real;
                imag[odd] = even_imag - product_imag;
            }
        }
    }
}

static void print_u64_decimal(uint64_t value)
{
    char digits[20];
    unsigned count = 0;
    do {
        digits[count++] = (char)('0' + value % 10u);
        value /= 10u;
    } while (value != 0u);
    while (count > 0u) {
        printf("%c", digits[--count]);
    }
}

int fft_cpu_testbench_run(void)
{
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch)
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            actual_real[batch][index] = input_real[index];
            actual_imag[batch][index] = input_imag[index];
        }

    uint64_t start = read_cycles();
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch)
        fft_cpu_f32(actual_real[batch], actual_imag[batch], FFT_SIZE);
    uint64_t end = read_cycles();
    uint64_t fft_cycles = end - start;

    float max_error = 0.0f;
    size_t max_error_bin = 0, max_error_batch = 0;
    float max_error_ratio = 0.0f;
    size_t max_error_ratio_bin = 0, max_error_ratio_batch = 0;
    size_t mismatch_count = 0, reported_mismatch_count = 0;
    size_t non_finite_count = 0;
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch) {
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            float ar = actual_real[batch][index];
            float ai = actual_imag[batch][index];
            if (!isfinite(ar) || !isfinite(ai)) {
                ++non_finite_count;
                ++mismatch_count;
                if (reported_mismatch_count < FFT_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u: non-finite actual\n",
                        (unsigned)batch, (unsigned)index);
                    ++reported_mismatch_count;
                }
                continue;
            }
            float real_error = fabsf(ar - groundtruth_real[index]);
            float imag_error = fabsf(ai - groundtruth_imag[index]);
            float error = real_error > imag_error ? real_error : imag_error;
            float real_limit = FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[index]);
            float imag_limit = FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[index]);
            float real_ratio = real_error / real_limit;
            float imag_ratio = imag_error / imag_limit;
            float error_ratio = real_ratio > imag_ratio ? real_ratio : imag_ratio;
            if (error > max_error) {
                max_error = error;
                max_error_bin = index;
                max_error_batch = batch;
            }
            if (error_ratio > max_error_ratio) {
                max_error_ratio = error_ratio;
                max_error_ratio_bin = index;
                max_error_ratio_batch = batch;
            }
            if (error_ratio > 1.0f) {
                ++mismatch_count;
                if (reported_mismatch_count < FFT_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u\n", (unsigned)batch,
                        (unsigned)index);
                    printf("  actual:   %.6f %c%.6fj\n", ar,
                        ai >= 0.0f ? '+' : '-', fabsf(ai));
                    printf("  expected: %.6f %c%.6fj\n", groundtruth_real[index],
                        groundtruth_imag[index] >= 0.0f ? '+' : '-',
                        fabsf(groundtruth_imag[index]));
                    ++reported_mismatch_count;
                }
            }
        }
    }

    printf("batches: %u\nFFT cycles: ", FFT_BATCH_COUNT);
    print_u64_decimal(fft_cycles);
    printf("\nFFT cycles per FFT: ");
    print_u64_decimal(fft_cycles / FFT_BATCH_COUNT);
    printf("\nmax component error: %.6f at batch %u bin %u\n", max_error,
        (unsigned)max_error_batch, (unsigned)max_error_bin);
    printf("max tolerance ratio: %.6f at batch %u bin %u\n", max_error_ratio,
        (unsigned)max_error_ratio_batch, (unsigned)max_error_ratio_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatch_count,
        (unsigned)(FFT_BATCH_COUNT * FFT_SIZE));
    if (mismatch_count > reported_mismatch_count)
        printf("mismatch report truncated: showed first %u of %u points\n",
            (unsigned)reported_mismatch_count, (unsigned)mismatch_count);
    printf("non-finite outputs: %u\n", (unsigned)non_finite_count);
    if (non_finite_count > 0 || max_error_ratio > 1.0f) {
        printf("FFT CPU: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        return 1;
    }
    printf("FFT CPU: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_cpu_testbench_run();
}
#endif
