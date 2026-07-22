/* SPDX-License-Identifier: Apache-2.0 */
/* Scalar reference for the batched radix-2 scaled Q7 FFT. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_cpu_int8_runner.h"
#include "fft_int8_vectors.h"
#include "nn_runtime.h"

#define FFT_CPU_INT8_BATCH_COUNT 64
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

static void fft_cpu_q7(int8_t *real,
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

    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;
        for (size_t block = 0; block < plan->size; block += block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                size_t even = block + offset;
                size_t odd = even + half;
                int32_t er = real[even], ei = imag[even];
                int32_t or_ = real[odd], oi = imag[odd];
                int32_t tr, ti;
                if (offset == 0) {
                    tr = or_;
                    ti = oi;
                } else {
                    int32_t wr = plan->twiddle_real[stage_base + offset];
                    int32_t wi = plan->twiddle_imag[stage_base + offset];
                    tr = saturate_i8(rnu_shift(or_ * wr - oi * wi, 7));
                    ti = saturate_i8(rnu_shift(oi * wr + or_ * wi, 7));
                }
                real[even] = (int8_t)rnu_shift(er + tr, 1);
                imag[even] = (int8_t)rnu_shift(ei + ti, 1);
                real[odd] = (int8_t)rnu_shift(er - tr, 1);
                imag[odd] = (int8_t)rnu_shift(ei - ti, 1);
            }
        }
    }
}

int fft_cpu_int8_testbench_run(void)
{
    fft_cpu_plan_q7_t plan;
    uint64_t plan_start = nn_runtime_read_cycles();
    int plan_ok = fft_cpu_plan_init_q7(
        &plan, FFT_INT8_SIZE, FFT_CPU_INT8_BATCH_COUNT);
    uint64_t plan_end = nn_runtime_read_cycles();
    if (!plan_ok) {
        printf("FFT CPU int8: FAIL (invalid plan)\n");
        return 1;
    }

    uint64_t layout_start = nn_runtime_read_cycles();
    fft_cpu_plan_load_input_q7(&plan);
    uint64_t layout_end = nn_runtime_read_cycles();
    uint64_t fft_start = nn_runtime_read_cycles();
    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        size_t base = batch * plan.size;
        fft_cpu_q7(&plan.data_real[base], &plan.data_imag[base], &plan);
    }
    uint64_t fft_end = nn_runtime_read_cycles();

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

    uint64_t plan_cycles = plan_end - plan_start;
    uint64_t layout_cycles = layout_end - layout_start;
    uint64_t fft_cycles = fft_end - fft_start;
    uint64_t reused_cycles = layout_cycles + fft_cycles;
    uint64_t first_cycles = plan_cycles + reused_cycles;
    printf("FFT size: %u\nstages: %u\nbatches: %u\nscale: 1/%u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count,
        (unsigned)plan.size);
    printf("twiddle plan cycles: "); print_u64_decimal(plan_cycles);
    printf("\ninput layout cycles: "); print_u64_decimal(layout_cycles);
    printf("\nFFT cycles: "); print_u64_decimal(fft_cycles);
    printf("\nreused-plan total cycles: "); print_u64_decimal(reused_cycles);
    printf("\nfirst-run total cycles: "); print_u64_decimal(first_cycles);
    printf("\nFFT cycles per FFT: "); print_u64_decimal(fft_cycles / plan.batch_count);
    printf("\nreused-plan cycles per FFT: "); print_u64_decimal(reused_cycles / plan.batch_count);
    printf("\nfirst-run cycles per FFT: "); print_u64_decimal(first_cycles / plan.batch_count);
    printf("\nmax component error: %d at batch %u bin %u\n", max_error,
        (unsigned)max_batch, (unsigned)max_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatches,
        (unsigned)(plan.size * plan.batch_count));
    if (mismatches > reported) printf("mismatch report truncated: showed first %u of %u points\n",
        (unsigned)reported, (unsigned)mismatches);

    if (mismatches != 0) {
        printf("FFT CPU int8: FAIL (bit-exact Q7 validation)\n");
        fft_cpu_plan_destroy_q7(&plan);
        return 1;
    }
    printf("FFT CPU int8: PASS (bit-exact Q7 validation)\n");
    fft_cpu_plan_destroy_q7(&plan);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_cpu_int8_testbench_run();
}
#endif
