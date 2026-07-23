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
#define FFT_BENCHMARK_RUNS 10
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

static void fft_cpu_bit_reverse_f32(float *real,
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
}

static void fft_cpu_butterflies_f32(float *real,
                                    float *imag,
                                    const fft_cpu_plan_f32_t *plan)
{
    size_t size = plan->size;
    for (size_t block_size = 2; block_size <= size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;

        for (size_t block = 0; block < size; block += block_size) {
            size_t odd = block + half;
            float odd_real = real[odd];
            float odd_imag = imag[odd];
            float even_real = real[block];
            float even_imag = imag[block];
            real[block] = even_real + odd_real;
            imag[block] = even_imag + odd_imag;
            real[odd] = even_real - odd_real;
            imag[odd] = even_imag - odd_imag;
        }

        for (size_t block = 0; block < size; block += block_size) {
            for (size_t offset = 1; offset < half; ++offset) {
                size_t even = block + offset;
                size_t odd = even + half;
                float odd_real = real[odd];
                float odd_imag = imag[odd];
                float even_real = real[even];
                float even_imag = imag[even];

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

static void fft_cpu_butterflies_stage_fused_f32(
    float *real, float *imag, const fft_cpu_plan_f32_t *plan)
{
    /*
     * Fuse two consecutive radix-2 stages. Four complex values are loaded,
     * both stages are evaluated through scalar temporaries, and only the
     * second-stage results are written back.
     */
    for (size_t block_size = 2; block_size <= plan->size;
         block_size <<= 2) {
        size_t half = block_size >> 1;
        size_t next_block_size = block_size << 1;
        size_t stage_a_base = half - 1;
        size_t stage_b_base = block_size - 1;

        for (size_t block = 0; block < plan->size;
             block += next_block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                size_t p0 = block + offset;
                size_t p1 = p0 + half;
                size_t p2 = p0 + block_size;
                size_t p3 = p1 + block_size;

                float r0 = real[p0], i0 = imag[p0];
                float r1 = real[p1], i1 = imag[p1];
                float r2 = real[p2], i2 = imag[p2];
                float r3 = real[p3], i3 = imag[p3];

                float ar = plan->twiddle_real[stage_a_base + offset];
                float ai = plan->twiddle_imag[stage_a_base + offset];
                float t1r = ar * r1 - ai * i1;
                float t1i = ar * i1 + ai * r1;
                float t3r = ar * r3 - ai * i3;
                float t3i = ar * i3 + ai * r3;
                float y0r = r0 + t1r, y0i = i0 + t1i;
                float y1r = r0 - t1r, y1i = i0 - t1i;
                float y2r = r2 + t3r, y2i = i2 + t3i;
                float y3r = r2 - t3r, y3i = i2 - t3i;

                float b0r = plan->twiddle_real[stage_b_base + offset];
                float b0i = plan->twiddle_imag[stage_b_base + offset];
                float b1r =
                    plan->twiddle_real[stage_b_base + half + offset];
                float b1i =
                    plan->twiddle_imag[stage_b_base + half + offset];
                float q0r = b0r * y2r - b0i * y2i;
                float q0i = b0r * y2i + b0i * y2r;
                float q1r = b1r * y3r - b1i * y3i;
                float q1i = b1r * y3i + b1i * y3r;

                real[p0] = y0r + q0r;
                imag[p0] = y0i + q0i;
                real[p1] = y1r + q1r;
                imag[p1] = y1i + q1i;
                real[p2] = y0r - q0r;
                imag[p2] = y0i - q0i;
                real[p3] = y1r - q1r;
                imag[p3] = y1i - q1i;
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

typedef void (*fft_cpu_butterfly_fn_f32)(
    float *, float *, const fft_cpu_plan_f32_t *);

static size_t count_mismatches_f32(const fft_cpu_plan_f32_t *plan)
{
    size_t count = 0;
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t index = 0; index < plan->size; ++index) {
            float ar = plan->data_real[base + index];
            float ai = plan->data_imag[base + index];
            float real_error = fabsf(ar - groundtruth_real[index]);
            float imag_error = fabsf(ai - groundtruth_imag[index]);
            float real_limit =
                FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[index]);
            float imag_limit =
                FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[index]);
            if (!isfinite(ar) || !isfinite(ai) ||
                real_error > real_limit || imag_error > imag_limit)
                ++count;
        }
    }
    return count;
}

static int run_timing_variant_f32(fft_cpu_plan_f32_t *plan,
                                  const char *name,
                                  uint64_t plan_cycles,
                                  fft_cpu_butterfly_fn_f32 function)
{
    uint64_t layout_cycles = 0;
    uint64_t bit_reverse_cycles = 0;
    uint64_t butterfly_cycles = 0;

    for (unsigned run = 0; run < FFT_BENCHMARK_RUNS; ++run) {
        uint64_t layout_start = nn_runtime_read_cycles();
        fft_cpu_plan_load_input_f32(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t base = batch * plan->size;
            fft_cpu_bit_reverse_f32(
                &plan->data_real[base], &plan->data_imag[base], plan);
        }
        uint64_t reverse_end = nn_runtime_read_cycles();
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t base = batch * plan->size;
            function(&plan->data_real[base], &plan->data_imag[base], plan);
        }
        uint64_t fft_end = nn_runtime_read_cycles();
        layout_cycles += layout_end - layout_start;
        bit_reverse_cycles += reverse_end - layout_end;
        butterfly_cycles += fft_end - reverse_end;
    }

    layout_cycles /= FFT_BENCHMARK_RUNS;
    bit_reverse_cycles /= FFT_BENCHMARK_RUNS;
    butterfly_cycles /= FFT_BENCHMARK_RUNS;
    uint64_t fft_cycles = bit_reverse_cycles + butterfly_cycles;
    uint64_t reused_cycles = layout_cycles + fft_cycles;
    size_t mismatches = count_mismatches_f32(plan);

    printf("\nCPU variant: %s\nbenchmark runs: %u", name,
        FFT_BENCHMARK_RUNS);
    printf("\naverage input layout cycles: ");
    print_u64_decimal(layout_cycles);
    printf("\naverage bit reversal cycles: ");
    print_u64_decimal(bit_reverse_cycles);
    printf("\naverage butterfly cycles: ");
    print_u64_decimal(butterfly_cycles);
    printf("\naverage FFT cycles: ");
    print_u64_decimal(fft_cycles);
    printf("\naverage reused-plan total cycles: ");
    print_u64_decimal(reused_cycles);
    printf("\nfirst-run equivalent cycles: ");
    print_u64_decimal(plan_cycles + reused_cycles);
    printf("\naverage butterfly cycles per FFT: ");
    print_u64_decimal(butterfly_cycles / plan->batch_count);
    printf("\naverage FFT cycles per FFT: ");
    print_u64_decimal(fft_cycles / plan->batch_count);
    printf("\nmismatched complex points: %u / %u\n",
        (unsigned)mismatches,
        (unsigned)(plan->batch_count * plan->size));
    printf("FFT CPU %s: %s\n", name, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
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

    uint64_t plan_cycles = plan_end - plan_start;
    printf("FFT size: %u\nstages: %u\nbatches: %u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count);
    printf("twiddle plan cycles: ");
    print_u64_decimal(plan_cycles);
    printf("\n");

    int baseline_ok = run_timing_variant_f32(
        &plan, "baseline", plan_cycles, fft_cpu_butterflies_f32);
    int fused_ok = 0;
    if ((plan.stage_count & 1u) == 0u) {
        fused_ok = run_timing_variant_f32(
            &plan, "stage-fused", plan_cycles,
            fft_cpu_butterflies_stage_fused_f32);
    } else {
        printf("FFT CPU stage-fused: FAIL (requires even stage count)\n");
    }

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
    if (!baseline_ok || !fused_ok || non_finite_count > 0 ||
        max_error_ratio > 1.0f) {
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
