/* SPDX-License-Identifier: Apache-2.0 */
/* Scalar radix-2 FFT using the same vectors and tolerances as the RVV test. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"
#include "nn_runtime.h"

#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_BATCH_COUNT 64
#define FFT_MAX_MISMATCH_REPORTS 64

typedef struct {
    size_t size;
    size_t batch_count;
    unsigned stage_count;
    float *data_real;
    float *data_imag;
    float *twiddle_real;
    float *twiddle_imag;
} fft_cpu_plan_f32_t;

static unsigned reverse_bits(unsigned value, unsigned bits)
{
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

static int fft_cpu_plan_init_f32(fft_cpu_plan_f32_t *plan,
                                 size_t size,
                                 size_t batch_count)
{
    if (size < 2 || size > FFT_SIZE || (size & (size - 1)) != 0 ||
        batch_count == 0 || batch_count > FFT_BATCH_COUNT) {
        return 0;
    }

    plan->size = size;
    plan->batch_count = batch_count;
    plan->stage_count = 0;
    plan->data_real = safe_malloc(size * batch_count * sizeof(float));
    plan->data_imag = safe_malloc(size * batch_count * sizeof(float));
    plan->twiddle_real = safe_malloc((size - 1) * sizeof(float));
    plan->twiddle_imag = safe_malloc((size - 1) * sizeof(float));

    for (size_t block_size = 2; block_size <= size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;
        ++plan->stage_count;
        for (size_t offset = 0; offset < half; ++offset) {
            float angle = -2.0f * (float)M_PI * (float)offset /
                (float)block_size;
            plan->twiddle_real[stage_base + offset] =
                offset == 0 ? 1.0f : cosf(angle);
            plan->twiddle_imag[stage_base + offset] =
                offset == 0 ? 0.0f : sinf(angle);
        }
    }
    return 1;
}

static void fft_cpu_plan_destroy_f32(fft_cpu_plan_f32_t *plan)
{
    safe_free(plan->twiddle_imag);
    safe_free(plan->twiddle_real);
    safe_free(plan->data_imag);
    safe_free(plan->data_real);
}

static void fft_cpu_plan_load_input_f32(const fft_cpu_plan_f32_t *plan)
{
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t index = 0; index < plan->size; ++index) {
            plan->data_real[base + index] = input_real[index];
            plan->data_imag[base + index] = input_imag[index];
        }
    }
}

static void fft_cpu_f32(float *real,
                        float *imag,
                        const fft_cpu_plan_f32_t *plan)
{
    size_t size = plan->size;

    for (size_t index = 0; index < size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, plan->stage_count);
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
        size_t stage_base = half - 1;
        for (size_t block = 0; block < size; block += block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                size_t even = block + offset;
                size_t odd = even + half;
                float odd_real = real[odd];
                float odd_imag = imag[odd];
                float even_real = real[even];
                float even_imag = imag[even];

                if (offset == 0) {
                    real[even] = even_real + odd_real;
                    imag[even] = even_imag + odd_imag;
                    real[odd] = even_real - odd_real;
                    imag[odd] = even_imag - odd_imag;
                    continue;
                }

                float wr = plan->twiddle_real[stage_base + offset];
                float wi = plan->twiddle_imag[stage_base + offset];
                float product_real = wr * odd_real - wi * odd_imag;
                float product_imag = wr * odd_imag + wi * odd_real;

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
    fft_cpu_plan_f32_t plan;
    uint64_t plan_start = nn_runtime_read_cycles();
    int plan_ok = fft_cpu_plan_init_f32(&plan, FFT_SIZE, FFT_BATCH_COUNT);
    uint64_t plan_end = nn_runtime_read_cycles();
    if (!plan_ok) {
        printf("FFT CPU: FAIL (invalid FFT plan)\n");
        return 1;
    }

    uint64_t layout_start = nn_runtime_read_cycles();
    fft_cpu_plan_load_input_f32(&plan);
    uint64_t layout_end = nn_runtime_read_cycles();
    uint64_t fft_start = nn_runtime_read_cycles();
    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        size_t base = batch * plan.size;
        fft_cpu_f32(&plan.data_real[base], &plan.data_imag[base], &plan);
    }
    uint64_t fft_end = nn_runtime_read_cycles();
    uint64_t plan_cycles = plan_end - plan_start;
    uint64_t layout_cycles = layout_end - layout_start;
    uint64_t fft_cycles = fft_end - fft_start;

    float max_error = 0.0f;
    size_t max_error_bin = 0, max_error_batch = 0;
    float max_error_ratio = 0.0f;
    size_t max_error_ratio_bin = 0, max_error_ratio_batch = 0;
    size_t mismatch_count = 0, reported_mismatch_count = 0;
    size_t non_finite_count = 0;
    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        size_t base = batch * plan.size;
        for (size_t index = 0; index < plan.size; ++index) {
            float ar = plan.data_real[base + index];
            float ai = plan.data_imag[base + index];
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

    uint64_t reused_total_cycles = layout_cycles + fft_cycles;
    uint64_t first_run_total_cycles = plan_cycles + reused_total_cycles;
    printf("FFT size: %u\nstages: %u\nbatches: %u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count);
    printf("twiddle plan cycles: ");
    print_u64_decimal(plan_cycles);
    printf("\ninput layout cycles: ");
    print_u64_decimal(layout_cycles);
    printf("\nFFT cycles: ");
    print_u64_decimal(fft_cycles);
    printf("\nreused-plan total cycles: ");
    print_u64_decimal(reused_total_cycles);
    printf("\nfirst-run total cycles: ");
    print_u64_decimal(first_run_total_cycles);
    printf("\nFFT cycles per FFT: ");
    print_u64_decimal(fft_cycles / plan.batch_count);
    printf("\nreused-plan cycles per FFT: ");
    print_u64_decimal(reused_total_cycles / plan.batch_count);
    printf("\nfirst-run cycles per FFT: ");
    print_u64_decimal(first_run_total_cycles / plan.batch_count);
    printf("\nmax component error: %.6f at batch %u bin %u\n", max_error,
        (unsigned)max_error_batch, (unsigned)max_error_bin);
    printf("max tolerance ratio: %.6f at batch %u bin %u\n", max_error_ratio,
        (unsigned)max_error_ratio_batch, (unsigned)max_error_ratio_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatch_count,
        (unsigned)(plan.batch_count * plan.size));
    if (mismatch_count > reported_mismatch_count)
        printf("mismatch report truncated: showed first %u of %u points\n",
            (unsigned)reported_mismatch_count, (unsigned)mismatch_count);
    printf("non-finite outputs: %u\n", (unsigned)non_finite_count);
    if (non_finite_count > 0 || max_error_ratio > 1.0f) {
        printf("FFT CPU: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        fft_cpu_plan_destroy_f32(&plan);
        return 1;
    }
    printf("FFT CPU: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    fft_cpu_plan_destroy_f32(&plan);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_cpu_testbench_run();
}
#endif
