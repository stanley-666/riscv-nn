/* SPDX-License-Identifier: Apache-2.0 */
/* Scalar reference for the batched radix-2 scaled Q7 FFT. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_cpu_int8_runner.h"
#include "fft_int8_vectors.h"
#include "../fft_batched_int8/fft_mixed_radix_q7.h"
#include "nn_runtime.h"

#define FFT_CPU_INT8_BATCH_COUNT 64
#define FFT_CPU_INT8_BENCHMARK_RUNS 10u
#define FFT_CPU_INT8_MAX_MISMATCH_REPORTS 64

typedef struct {
    size_t size;
    size_t batch_count;
    unsigned stage_count;
    int8_t *data_real;
    int8_t *data_imag;
    int8_t *twiddle_real;
    int8_t *twiddle_imag;
} fft_cpu_plan_q7_t;

static void print_u64_decimal(uint64_t value)
{
    char digits[20];
    unsigned count = 0;
    do {
        digits[count++] = (char)('0' + value % 10u);
        value /= 10u;
    } while (value != 0u);
    while (count > 0u) printf("%c", digits[--count]);
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

static int32_t rnu_shift(int32_t value, unsigned shift)
{
    return (value + (INT32_C(1) << (shift - 1))) >> shift;
}

static int8_t saturate_i8(int32_t value)
{
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static int8_t quantize_q7(float value)
{
    long scaled = lroundf(value * 128.0f);
    if (scaled > 127) scaled = 127;
    if (scaled < -128) scaled = -128;
    return (int8_t)scaled;
}

static int fft_cpu_plan_init_q7(fft_cpu_plan_q7_t *plan,
                                size_t size,
                                size_t batch_count)
{
    if (size < 2 || size > FFT_INT8_SIZE || (size & (size - 1)) != 0 ||
        batch_count == 0 || batch_count > FFT_CPU_INT8_BATCH_COUNT) return 0;
    plan->size = size;
    plan->batch_count = batch_count;
    plan->stage_count = 0;
    plan->data_real = safe_malloc(size * batch_count);
    plan->data_imag = safe_malloc(size * batch_count);
    plan->twiddle_real = safe_malloc(size - 1);
    plan->twiddle_imag = safe_malloc(size - 1);

    for (size_t block_size = 2; block_size <= size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;
        ++plan->stage_count;
        for (size_t offset = 0; offset < half; ++offset) {
            if (offset == 0) {
                plan->twiddle_real[stage_base] = 127;
                plan->twiddle_imag[stage_base] = 0;
            } else {
                float angle = -2.0f * (float)M_PI * (float)offset /
                    (float)block_size;
                plan->twiddle_real[stage_base + offset] = quantize_q7(cosf(angle));
                plan->twiddle_imag[stage_base + offset] = quantize_q7(sinf(angle));
            }
        }
    }
    return 1;
}

static void fft_cpu_plan_destroy_q7(fft_cpu_plan_q7_t *plan)
{
    safe_free(plan->twiddle_imag);
    safe_free(plan->twiddle_real);
    safe_free(plan->data_imag);
    safe_free(plan->data_real);
}

static void fft_cpu_plan_load_input_q7(const fft_cpu_plan_q7_t *plan)
{
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t index = 0; index < plan->size; ++index) {
            plan->data_real[base + index] = fft_int8_input_real[index];
            plan->data_imag[base + index] = fft_int8_input_imag[index];
        }
    }
}

static void fft_cpu_bit_reverse_q7(int8_t *real,
                                   int8_t *imag,
                                   const fft_cpu_plan_q7_t *plan)
{
    for (size_t index = 0; index < plan->size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, plan->stage_count);
        if (reversed > index) {
            int8_t temporary = real[index];
            real[index] = real[reversed];
            real[reversed] = temporary;
            temporary = imag[index];
            imag[index] = imag[reversed];
            imag[reversed] = temporary;
        }
    }
}

static void fft_cpu_butterflies_q7(int8_t *real,
                                   int8_t *imag,
                                   const fft_cpu_plan_q7_t *plan)
{
    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;

        for (size_t block = 0; block < plan->size; block += block_size) {
            size_t odd = block + half;
            int32_t er = real[block], ei = imag[block];
            int32_t or_ = real[odd], oi = imag[odd];
            real[block] = (int8_t)rnu_shift(er + or_, 1);
            imag[block] = (int8_t)rnu_shift(ei + oi, 1);
            real[odd] = (int8_t)rnu_shift(er - or_, 1);
            imag[odd] = (int8_t)rnu_shift(ei - oi, 1);
        }

        for (size_t block = 0; block < plan->size; block += block_size) {
            for (size_t offset = 1; offset < half; ++offset) {
                size_t even = block + offset;
                size_t odd = even + half;
                int32_t er = real[even], ei = imag[even];
                int32_t or_ = real[odd], oi = imag[odd];
                int32_t wr = plan->twiddle_real[stage_base + offset];
                int32_t wi = plan->twiddle_imag[stage_base + offset];
                int32_t tr = saturate_i8(rnu_shift(or_ * wr - oi * wi, 7));
                int32_t ti = saturate_i8(rnu_shift(oi * wr + or_ * wi, 7));
                real[even] = (int8_t)rnu_shift(er + tr, 1);
                imag[even] = (int8_t)rnu_shift(ei + ti, 1);
                real[odd] = (int8_t)rnu_shift(er - tr, 1);
                imag[odd] = (int8_t)rnu_shift(ei - ti, 1);
            }
        }
    }
}

static void scalar_butterfly_q7(int8_t er, int8_t ei,
                                int8_t or_, int8_t oi,
                                int8_t wr, int8_t wi, int unity,
                                int8_t *upper_r, int8_t *upper_i,
                                int8_t *lower_r, int8_t *lower_i)
{
    int32_t tr = or_;
    int32_t ti = oi;
    if (!unity) {
        tr = saturate_i8(rnu_shift(
            (int32_t)or_ * wr - (int32_t)oi * wi, 7));
        ti = saturate_i8(rnu_shift(
            (int32_t)oi * wr + (int32_t)or_ * wi, 7));
    }
    *upper_r = (int8_t)rnu_shift((int32_t)er + tr, 1);
    *upper_i = (int8_t)rnu_shift((int32_t)ei + ti, 1);
    *lower_r = (int8_t)rnu_shift((int32_t)er - tr, 1);
    *lower_i = (int8_t)rnu_shift((int32_t)ei - ti, 1);
}

static void fft_cpu_butterflies_stage_fused_q7(
    int8_t *real, int8_t *imag, const fft_cpu_plan_q7_t *plan)
{
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
                int8_t y0r, y0i, y1r, y1i;
                int8_t y2r, y2i, y3r, y3i;
                int8_t z0r, z0i, z2r, z2i;
                int8_t z1r, z1i, z3r, z3i;
                int8_t ar = plan->twiddle_real[stage_a_base + offset];
                int8_t ai = plan->twiddle_imag[stage_a_base + offset];
                int8_t b0r = plan->twiddle_real[stage_b_base + offset];
                int8_t b0i = plan->twiddle_imag[stage_b_base + offset];
                int8_t b1r =
                    plan->twiddle_real[stage_b_base + half + offset];
                int8_t b1i =
                    plan->twiddle_imag[stage_b_base + half + offset];

                scalar_butterfly_q7(
                    real[p0], imag[p0], real[p1], imag[p1], ar, ai,
                    offset == 0, &y0r, &y0i, &y1r, &y1i);
                scalar_butterfly_q7(
                    real[p2], imag[p2], real[p3], imag[p3], ar, ai,
                    offset == 0, &y2r, &y2i, &y3r, &y3i);
                scalar_butterfly_q7(
                    y0r, y0i, y2r, y2i, b0r, b0i, offset == 0,
                    &z0r, &z0i, &z2r, &z2i);
                scalar_butterfly_q7(
                    y1r, y1i, y3r, y3i, b1r, b1i, 0,
                    &z1r, &z1i, &z3r, &z3i);
                real[p0] = z0r; imag[p0] = z0i;
                real[p1] = z1r; imag[p1] = z1i;
                real[p2] = z2r; imag[p2] = z2i;
                real[p3] = z3r; imag[p3] = z3i;
            }
        }
    }
}

typedef void (*fft_cpu_q7_function_t)(
    int8_t *, int8_t *, const fft_cpu_plan_q7_t *);

static size_t count_mismatches_q7(const fft_cpu_plan_q7_t *plan)
{
    size_t mismatches = 0;
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t bin = 0; bin < plan->size; ++bin)
            if (plan->data_real[base + bin] !=
                    fft_int8_groundtruth_real[bin] ||
                plan->data_imag[base + bin] !=
                    fft_int8_groundtruth_imag[bin])
                ++mismatches;
    }
    return mismatches;
}

static int run_cpu_q7_variant(fft_cpu_plan_q7_t *plan, const char *name,
                              uint64_t plan_cycles,
                              fft_cpu_q7_function_t function)
{
    uint64_t layout = 0, reverse = 0, butterfly = 0;
    for (unsigned run = 0; run < FFT_CPU_INT8_BENCHMARK_RUNS; ++run) {
        uint64_t begin = nn_runtime_read_cycles();
        fft_cpu_plan_load_input_q7(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t base = batch * plan->size;
            fft_cpu_bit_reverse_q7(
                &plan->data_real[base], &plan->data_imag[base], plan);
        }
        uint64_t reverse_end = nn_runtime_read_cycles();
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t base = batch * plan->size;
            function(&plan->data_real[base], &plan->data_imag[base], plan);
        }
        uint64_t end = nn_runtime_read_cycles();
        layout += layout_end - begin;
        reverse += reverse_end - layout_end;
        butterfly += end - reverse_end;
    }
    layout /= FFT_CPU_INT8_BENCHMARK_RUNS;
    reverse /= FFT_CPU_INT8_BENCHMARK_RUNS;
    butterfly /= FFT_CPU_INT8_BENCHMARK_RUNS;
    uint64_t fft = reverse + butterfly;
    uint64_t reused = layout + fft;
    size_t mismatches = count_mismatches_q7(plan);
    printf("\nCPU int8 variant: %s\nbenchmark runs: %u\n", name,
        FFT_CPU_INT8_BENCHMARK_RUNS);
    printf("average input layout cycles: "); print_u64_decimal(layout);
    printf("\naverage bit reversal cycles: "); print_u64_decimal(reverse);
    printf("\naverage butterfly cycles: "); print_u64_decimal(butterfly);
    printf("\naverage FFT cycles: "); print_u64_decimal(fft);
    printf("\naverage reused-plan total cycles: "); print_u64_decimal(reused);
    printf("\nfirst-run equivalent cycles: ");
    print_u64_decimal(plan_cycles + reused);
    printf("\naverage butterfly cycles per FFT: ");
    print_u64_decimal(butterfly / plan->batch_count);
    printf("\naverage FFT cycles per FFT: ");
    print_u64_decimal(fft / plan->batch_count);
    printf("\nmismatched complex points: %u / %u\n",
        (unsigned)mismatches,
        (unsigned)(plan->size * plan->batch_count));
    printf("FFT CPU int8 %s: %s\n", name,
        mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

static int run_cpu_mixed_radix_variant(
    fft_cpu_plan_q7_t *plan, const fft_mixed_q7_plan_t *mixed_plan)
{
    int8_t reference_real[FFT_INT8_SIZE];
    int8_t reference_imag[FFT_INT8_SIZE];
    for (size_t bin = 0; bin < plan->size; ++bin) {
        reference_real[bin] = fft_int8_input_real[bin];
        reference_imag[bin] = fft_int8_input_imag[bin];
    }
    fft_mixed_q7_scalar(
        mixed_plan, reference_real, reference_imag, plan->size);

    uint64_t layout = 0, fft = 0;
    for (unsigned run = 0; run < FFT_CPU_INT8_BENCHMARK_RUNS; ++run) {
        uint64_t begin = nn_runtime_read_cycles();
        fft_cpu_plan_load_input_q7(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t base = batch * plan->size;
            fft_mixed_q7_scalar(mixed_plan, &plan->data_real[base],
                &plan->data_imag[base], plan->size);
        }
        uint64_t end = nn_runtime_read_cycles();
        layout += layout_end - begin;
        fft += end - layout_end;
    }
    layout /= FFT_CPU_INT8_BENCHMARK_RUNS;
    fft /= FFT_CPU_INT8_BENCHMARK_RUNS;

    size_t reference_mismatches = 0, radix2_differences = 0;
    size_t within_one = 0;
    uint64_t absolute_error_sum = 0;
    unsigned max_component_error = 0;
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t bin = 0; bin < plan->size; ++bin) {
            int8_t actual_real = plan->data_real[base + bin];
            int8_t actual_imag = plan->data_imag[base + bin];
            if (actual_real != reference_real[bin] ||
                actual_imag != reference_imag[bin])
                ++reference_mismatches;
            if (actual_real != fft_int8_groundtruth_real[bin] ||
                actual_imag != fft_int8_groundtruth_imag[bin])
                ++radix2_differences;
            int real_error =
                (int)actual_real - fft_int8_groundtruth_real[bin];
            int imag_error =
                (int)actual_imag - fft_int8_groundtruth_imag[bin];
            if (real_error < 0) real_error = -real_error;
            if (imag_error < 0) imag_error = -imag_error;
            absolute_error_sum += (unsigned)real_error + (unsigned)imag_error;
            if (real_error <= 1 && imag_error <= 1) ++within_one;
            if ((unsigned)real_error > max_component_error)
                max_component_error = (unsigned)real_error;
            if ((unsigned)imag_error > max_component_error)
                max_component_error = (unsigned)imag_error;
        }
    }

    printf("\nCPU int8 variant: mixed-radix-4-16-16\n");
    printf("benchmark runs: %u\n", FFT_CPU_INT8_BENCHMARK_RUNS);
    printf("dense transforms per FFT: %u\n", (unsigned)mixed_plan->count);
    printf("average input layout cycles: "); print_u64_decimal(layout);
    printf("\naverage FFT cycles: "); print_u64_decimal(fft);
    printf("\nmixed-radix-reference mismatched complex points: %u / %u\n",
        (unsigned)reference_mismatches,
        (unsigned)(plan->size * plan->batch_count));
    printf("radix-2-groundtruth differing complex points: %u / %u\n",
        (unsigned)radix2_differences,
        (unsigned)(plan->size * plan->batch_count));
    printf("radix-2-groundtruth points within +/-1 per component: %u / %u\n",
        (unsigned)within_one,
        (unsigned)(plan->size * plan->batch_count));
    printf("radix-2-groundtruth mean absolute component error x1000: %lu\n",
        (unsigned long)(absolute_error_sum * 1000 /
            (2 * plan->size * plan->batch_count)));
    printf("radix-2-groundtruth maximum component error: %u\n",
        max_component_error);
    printf("FFT CPU int8 mixed-radix-4-16-16: %s\n",
        reference_mismatches == 0 ? "PASS" : "FAIL");
    return reference_mismatches == 0;
}

int fft_cpu_int8_testbench_run(void)
{
    fft_cpu_plan_q7_t plan;
    static fft_mixed_q7_plan_t mixed_plan;
    uint64_t plan_start = nn_runtime_read_cycles();
    int plan_ok = fft_cpu_plan_init_q7(
        &plan, FFT_INT8_SIZE, FFT_CPU_INT8_BATCH_COUNT);
    fft_mixed_q7_plan_init(&mixed_plan);
    uint64_t plan_end = nn_runtime_read_cycles();
    if (!plan_ok) {
        printf("FFT CPU int8: FAIL (invalid plan)\n");
        return 1;
    }

    uint64_t plan_cycles = plan_end - plan_start;
    printf("FFT size: %u\nstages: %u\nbatches: %u\nscale: 1/%u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count,
        (unsigned)plan.size);
    printf("twiddle plan cycles: "); print_u64_decimal(plan_cycles);
    printf("\n");
    int baseline_ok = run_cpu_q7_variant(
        &plan, "baseline", plan_cycles, fft_cpu_butterflies_q7);
    int mixed_ok = run_cpu_mixed_radix_variant(&plan, &mixed_plan);
    int fused_ok = 0;
    if ((plan.stage_count & 1u) == 0u)
        fused_ok = run_cpu_q7_variant(
            &plan, "stage-fused", plan_cycles,
            fft_cpu_butterflies_stage_fused_q7);
    else
        printf("FFT CPU int8 stage-fused: FAIL (requires even stages)\n");

    size_t mismatches = 0, reported = 0;
    int max_error = 0;
    size_t max_batch = 0, max_bin = 0;
    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        size_t base = batch * plan.size;
        for (size_t bin = 0; bin < plan.size; ++bin) {
            int dr = (int)plan.data_real[base + bin] -
                (int)fft_int8_groundtruth_real[bin];
            int di = (int)plan.data_imag[base + bin] -
                (int)fft_int8_groundtruth_imag[bin];
            int error = dr < 0 ? -dr : dr;
            int imag_error = di < 0 ? -di : di;
            if (imag_error > error) error = imag_error;
            if (error > max_error) { max_error = error; max_batch = batch; max_bin = bin; }
            if (dr != 0 || di != 0) {
                ++mismatches;
                if (reported < FFT_CPU_INT8_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u: actual %d %dj expected %d %dj\n",
                        (unsigned)batch, (unsigned)bin,
                        (int)plan.data_real[base + bin],
                        (int)plan.data_imag[base + bin],
                        (int)fft_int8_groundtruth_real[bin],
                        (int)fft_int8_groundtruth_imag[bin]);
                    ++reported;
                }
            }
        }
    }

    printf("\nmax component error: %d at batch %u bin %u\n", max_error,
        (unsigned)max_batch, (unsigned)max_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatches,
        (unsigned)(plan.size * plan.batch_count));
    if (mismatches > reported) printf("mismatch report truncated: showed first %u of %u points\n",
        (unsigned)reported, (unsigned)mismatches);

    if (!baseline_ok || !mixed_ok || !fused_ok || mismatches != 0) {
        printf("FFT CPU int8: FAIL (bit-exact Q7 validation)\n");
        fft_cpu_plan_destroy_q7(&plan);
        return 1;
    }
    printf("FFT CPU int8: PASS (bit-exact Q7 validation)\n");
    fft_cpu_plan_destroy_q7(&plan);
    return 0;
}

int main(void)
{
    return fft_cpu_int8_testbench_run();
}
