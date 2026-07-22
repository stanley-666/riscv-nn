/* SPDX-License-Identifier: Apache-2.0 */
/* Batched radix-2 FFT with vectors spanning the batch dimension. */

#define _DEFAULT_SOURCE

#include <float.h>
#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"
#include "nn_runtime.h"

#define FFT_BATCH_COUNT 64
#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_MAX_MISMATCH_REPORTS 64

typedef struct {
    size_t size;
    size_t batch_count;
    unsigned stage_count;
    float *data_real;
    float *data_imag;
    float *twiddle_real;
    float *twiddle_imag;
} fft_plan_f32_t;

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

static inline size_t data_offset(const fft_plan_f32_t *plan,
                                 size_t index,
                                 size_t batch)
{
    return index * plan->batch_count + batch;
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

/* Keep the FPGA-compatible swap at e32,m1.  GCC may otherwise canonicalize
 * equivalent intrinsics to a different SEW/LMUL combination. */
static __attribute__((noinline)) int swap_rows_e32m1(float *a_real,
                                                     float *a_imag,
                                                     float *b_real,
                                                     float *b_imag,
                                                     size_t count)
{
    while (count != 0) {
        size_t vl;
        __asm__ volatile(
            "vsetvli %0, %1, e32, m1, ta, ma\n\t"
            "vle32.v v8, (%2)\n\t"
            "vle32.v v9, (%3)\n\t"
            "vle32.v v10, (%4)\n\t"
            "vle32.v v11, (%5)\n\t"
            "vse32.v v10, (%2)\n\t"
            "vse32.v v11, (%3)\n\t"
            "vse32.v v8, (%4)\n\t"
            "vse32.v v9, (%5)"
            : "=&r"(vl)
            : "r"(count), "r"(a_real), "r"(a_imag),
              "r"(b_real), "r"(b_imag)
            : "v8", "v9", "v10", "v11", "memory");
        if (vl == 0 || vl > count) return 0;
        a_real += vl;
        a_imag += vl;
        b_real += vl;
        b_imag += vl;
        count -= vl;
    }
    return 1;
}

static int fft_plan_init_f32(fft_plan_f32_t *plan,
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
    size_t data_elements = size * batch_count;
    size_t twiddle_elements = size - 1;
    plan->data_real = safe_malloc(data_elements * sizeof(float));
    plan->data_imag = safe_malloc(data_elements * sizeof(float));
    plan->twiddle_real = safe_malloc(twiddle_elements * sizeof(float));
    plan->twiddle_imag = safe_malloc(twiddle_elements * sizeof(float));

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

static void fft_plan_destroy_f32(fft_plan_f32_t *plan)
{
    safe_free(plan->twiddle_imag);
    safe_free(plan->twiddle_real);
    safe_free(plan->data_imag);
    safe_free(plan->data_real);
    plan->twiddle_imag = NULL;
    plan->twiddle_real = NULL;
    plan->data_imag = NULL;
    plan->data_real = NULL;
}

static int fft_plan_load_input_f32(const fft_plan_f32_t *plan)
{
    for (size_t index = 0; index < plan->size; ++index) {
        size_t row = index * plan->batch_count;
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            plan->data_real[row + batch] = input_real[index];
            plan->data_imag[row + batch] = input_imag[index];
        }
    }
    return 1;
}

static inline void butterfly_rvv_f32(float *restrict even_real,
                                     float *restrict even_imag,
                                     float *restrict odd_real,
                                     float *restrict odd_imag,
                                     float twiddle_real,
                                     float twiddle_imag,
                                     size_t vl)
{
    /* Four LMUL=m8 groups fill the 32-register vector file.  Issue the two
     * independent loads first, then reuse their groups after both complex
     * products have consumed them. */
    vfloat32m8_t odd_r = __riscv_vle32_v_f32m8(odd_real, vl);
    vfloat32m8_t odd_i = __riscv_vle32_v_f32m8(odd_imag, vl);

    vfloat32m8_t product_r = __riscv_vfmul_vf_f32m8(odd_r, twiddle_real, vl);
    vfloat32m8_t product_i = __riscv_vfmul_vf_f32m8(odd_i, twiddle_real, vl);
    product_r = __riscv_vfnmsac_vf_f32m8(product_r, twiddle_imag, odd_i, vl);
    product_i = __riscv_vfmacc_vf_f32m8(product_i, twiddle_imag, odd_r, vl);

    vfloat32m8_t even_r = __riscv_vle32_v_f32m8(even_real, vl);
    vfloat32m8_t upper_r = __riscv_vfadd_vv_f32m8(even_r, product_r, vl);
    vfloat32m8_t lower_r = __riscv_vfsub_vv_f32m8(even_r, product_r, vl);
    __riscv_vse32_v_f32m8(even_real, upper_r, vl);
    __riscv_vse32_v_f32m8(odd_real, lower_r, vl);

    vfloat32m8_t even_i = __riscv_vle32_v_f32m8(even_imag, vl);
    vfloat32m8_t upper_i = __riscv_vfadd_vv_f32m8(even_i, product_i, vl);
    vfloat32m8_t lower_i = __riscv_vfsub_vv_f32m8(even_i, product_i, vl);
    __riscv_vse32_v_f32m8(even_imag, upper_i, vl);
    __riscv_vse32_v_f32m8(odd_imag, lower_i, vl);
}

static inline void butterfly_unity_rvv_f32(float *restrict even_real,
                                           float *restrict even_imag,
                                           float *restrict odd_real,
                                           float *restrict odd_imag,
                                           size_t vl)
{
    vfloat32m8_t odd_r = __riscv_vle32_v_f32m8(odd_real, vl);
    vfloat32m8_t even_r = __riscv_vle32_v_f32m8(even_real, vl);
    vfloat32m8_t upper_r = __riscv_vfadd_vv_f32m8(even_r, odd_r, vl);
    vfloat32m8_t lower_r = __riscv_vfsub_vv_f32m8(even_r, odd_r, vl);
    __riscv_vse32_v_f32m8(even_real, upper_r, vl);
    __riscv_vse32_v_f32m8(odd_real, lower_r, vl);

    vfloat32m8_t odd_i = __riscv_vle32_v_f32m8(odd_imag, vl);
    vfloat32m8_t even_i = __riscv_vle32_v_f32m8(even_imag, vl);
    vfloat32m8_t upper_i = __riscv_vfadd_vv_f32m8(even_i, odd_i, vl);
    vfloat32m8_t lower_i = __riscv_vfsub_vv_f32m8(even_i, odd_i, vl);
    __riscv_vse32_v_f32m8(even_imag, upper_i, vl);
    __riscv_vse32_v_f32m8(odd_imag, lower_i, vl);
}

static inline int butterfly_batches_rvv_f32(float *even_real,
                                             float *even_imag,
                                             float *odd_real,
                                             float *odd_imag,
                                             float twiddle_real,
                                             float twiddle_imag,
                                             size_t batch_count,
                                             size_t vlmax)
{
    size_t batch = 0;
    for (; batch_count - batch >= vlmax; batch += vlmax) {
        butterfly_rvv_f32(even_real + batch, even_imag + batch,
                          odd_real + batch, odd_imag + batch,
                          twiddle_real, twiddle_imag, vlmax);
    }
    if (batch < batch_count) {
        size_t remaining = batch_count - batch;
        size_t tail_vl = __riscv_vsetvl_e32m8(remaining);
        if (tail_vl == 0 || tail_vl > remaining) return 0;
        butterfly_rvv_f32(even_real + batch, even_imag + batch,
                          odd_real + batch, odd_imag + batch,
                          twiddle_real, twiddle_imag, tail_vl);
    }
    return 1;
}

static inline int butterfly_unity_batches_rvv_f32(float *even_real,
                                                   float *even_imag,
                                                   float *odd_real,
                                                   float *odd_imag,
                                                   size_t batch_count,
                                                   size_t vlmax)
{
    size_t batch = 0;
    for (; batch_count - batch >= vlmax; batch += vlmax) {
        butterfly_unity_rvv_f32(even_real + batch, even_imag + batch,
                                odd_real + batch, odd_imag + batch, vlmax);
    }
    if (batch < batch_count) {
        size_t remaining = batch_count - batch;
        size_t tail_vl = __riscv_vsetvl_e32m8(remaining);
        if (tail_vl == 0 || tail_vl > remaining) return 0;
        butterfly_unity_rvv_f32(even_real + batch, even_imag + batch,
                                odd_real + batch, odd_imag + batch, tail_vl);
    }
    return 1;
}

static int fft_batched_rvv_f32(const fft_plan_f32_t *plan)
{
    for (size_t index = 0; index < plan->size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, plan->stage_count);
        if (reversed <= index) continue;
        size_t a = data_offset(plan, index, 0);
        size_t b = data_offset(plan, reversed, 0);
        if (!swap_rows_e32m1(&plan->data_real[a], &plan->data_imag[a],
                             &plan->data_real[b], &plan->data_imag[b],
                             plan->batch_count)) return 0;
    }

    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    if (vlmax == 0) return 0;

    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;
        for (size_t offset = 0; offset < half; ++offset) {
            float wr = plan->twiddle_real[stage_base + offset];
            float wi = plan->twiddle_imag[stage_base + offset];

            for (size_t block = 0; block < plan->size;
                 block += block_size) {
                size_t even = block + offset;
                size_t odd = even + half;
                size_t even_base = data_offset(plan, even, 0);
                size_t odd_base = data_offset(plan, odd, 0);
                int ok = offset == 0 ?
                    butterfly_unity_batches_rvv_f32(
                        &plan->data_real[even_base],
                        &plan->data_imag[even_base],
                        &plan->data_real[odd_base],
                        &plan->data_imag[odd_base],
                        plan->batch_count, vlmax) :
                    butterfly_batches_rvv_f32(
                        &plan->data_real[even_base],
                        &plan->data_imag[even_base],
                        &plan->data_real[odd_base],
                        &plan->data_imag[odd_base], wr, wi,
                        plan->batch_count, vlmax);
                if (!ok) return 0;
          }
        }
    }
    return 1;
}

static int input_has_safe_fp32_bound(void)
{
    float max_l1 = 0.0f;
    for (size_t index = 0; index < FFT_SIZE; ++index) {
        if (!isfinite(input_real[index]) || !isfinite(input_imag[index])) {
            printf("FFT input contains NaN or Inf at bin %u\n", (unsigned)index);
            return 0;
        }
        float l1 = fabsf(input_real[index]) + fabsf(input_imag[index]);
        if (l1 > max_l1) max_l1 = l1;
    }

    /* Every DFT output component is bounded by
     * FFT_SIZE * max(|real| + |imag|). Use double for the bound check itself. */
    if ((double)max_l1 * (double)FFT_SIZE > (double)FLT_MAX) {
        printf("FFT input may overflow FP32 during butterfly accumulation\n");
        return 0;
    }
    return 1;
}

int fft_batched_testbench_run(void)
{
    if (!input_has_safe_fp32_bound()) return 1;

    fft_plan_f32_t plan;
    uint64_t plan_start = nn_runtime_read_cycles();
    int plan_ok = fft_plan_init_f32(&plan, FFT_SIZE, FFT_BATCH_COUNT);
    uint64_t plan_end = nn_runtime_read_cycles();
    if (!plan_ok) {
        printf("FFT batched RVV: FAIL (invalid FFT plan)\n");
        return 1;
    }
    uint64_t layout_start = nn_runtime_read_cycles();
    int layout_vector_length_ok = fft_plan_load_input_f32(&plan);
    uint64_t layout_end = nn_runtime_read_cycles();
    uint64_t fft_start = nn_runtime_read_cycles();
    int vector_length_ok = fft_batched_rvv_f32(&plan);
    uint64_t fft_end = nn_runtime_read_cycles();
    uint64_t plan_cycles = plan_end - plan_start;
    uint64_t layout_cycles = layout_end - layout_start;
    uint64_t fft_cycles = fft_end - fft_start;
    float max_error = 0.0f, max_ratio = 0.0f;
    size_t max_bin = 0, max_batch = 0, ratio_bin = 0, ratio_batch = 0;
    size_t non_finite_count = 0, first_non_finite_bin = 0, first_non_finite_batch = 0;
    size_t mismatch_count = 0, reported_mismatch_count = 0;
    size_t batch_mismatch_count[FFT_BATCH_COUNT] = {0};

    for (size_t index = 0; index < plan.size; ++index) {
        for (size_t batch = 0; batch < plan.batch_count; ++batch) {
            size_t output = data_offset(&plan, index, batch);
            if (!isfinite(plan.data_real[output]) ||
                !isfinite(plan.data_imag[output])) {
                if (non_finite_count == 0) {
                    first_non_finite_bin = index;
                    first_non_finite_batch = batch;
                }
                ++non_finite_count;
                ++mismatch_count;
                ++batch_mismatch_count[batch];
                if (reported_mismatch_count < FFT_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u: non-finite actual\n",
                        (unsigned)batch, (unsigned)index);
                    printf("  expected: %.6f ", groundtruth_real[index]);
                    if (groundtruth_imag[index] >= 0.0f) printf("+");
                    printf("%.6fj\n", groundtruth_imag[index]);
                    ++reported_mismatch_count;
                }
                continue;
            }
            float re = fabsf(plan.data_real[output] - groundtruth_real[index]);
            float ie = fabsf(plan.data_imag[output] - groundtruth_imag[index]);
            float error = re > ie ? re : ie;
            float rr = re / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[index]));
            float ir = ie / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[index]));
            float ratio = rr > ir ? rr : ir;
            if (error > max_error) { max_error = error; max_bin = index; max_batch = batch; }
            if (ratio > max_ratio) { max_ratio = ratio; ratio_bin = index; ratio_batch = batch; }
            if (ratio > 1.0f) {
                ++mismatch_count;
                ++batch_mismatch_count[batch];
                if (reported_mismatch_count < FFT_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u\n", (unsigned)batch,
                        (unsigned)index);
                    printf("  actual:   %.6f ", plan.data_real[output]);
                    if (plan.data_imag[output] >= 0.0f) printf("+");
                    printf("%.6fj\n", plan.data_imag[output]);
                    printf("  expected: %.6f ", groundtruth_real[index]);
                    if (groundtruth_imag[index] >= 0.0f) printf("+");
                    printf("%.6fj\n", groundtruth_imag[index]);
                    printf("  error:    real %.6f imag %.6f\n", re, ie);
                    ++reported_mismatch_count;
                }
            }
        }
    }

    uint64_t reusable_total_cycles = layout_cycles + fft_cycles;
    uint64_t first_run_total_cycles = plan_cycles + reusable_total_cycles;
    printf("FFT size: %u\nstages: %u\nbatches: %u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count);
    printf("twiddle plan cycles: "); print_u64_decimal(plan_cycles);
    printf("\ninput layout cycles: "); print_u64_decimal(layout_cycles);
    printf("\nFFT cycles: "); print_u64_decimal(fft_cycles);
    printf("\nreused-plan total cycles: "); print_u64_decimal(reusable_total_cycles);
    printf("\nfirst-run total cycles: "); print_u64_decimal(first_run_total_cycles);
    printf("\nFFT cycles per FFT: "); print_u64_decimal(fft_cycles / plan.batch_count);
    printf("\nreused-plan cycles per FFT: ");
    print_u64_decimal(reusable_total_cycles / plan.batch_count);
    printf("\nfirst-run cycles per FFT: ");
    print_u64_decimal(first_run_total_cycles / plan.batch_count);
    printf("\nmax component error: %.6f at batch %u bin %u\n", max_error, (unsigned)max_batch, (unsigned)max_bin);
    printf("max tolerance ratio: %.6f at batch %u bin %u\n", max_ratio, (unsigned)ratio_batch, (unsigned)ratio_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatch_count,
        (unsigned)(plan.batch_count * plan.size));
    if (mismatch_count > reported_mismatch_count) {
        printf("mismatch report truncated: showed first %u of %u points\n",
            (unsigned)reported_mismatch_count, (unsigned)mismatch_count);
    }
    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        if (batch_mismatch_count[batch] != 0) {
            printf("batch %u mismatches: %u / %u\n", (unsigned)batch,
                (unsigned)batch_mismatch_count[batch], (unsigned)plan.size);
        }
    }
    printf("non-finite outputs: %u\n", (unsigned)non_finite_count);
    if (non_finite_count > 0) {
        printf("first non-finite output: batch %u bin %u\n",
            (unsigned)first_non_finite_batch, (unsigned)first_non_finite_bin);
    }
    if (!layout_vector_length_ok || !vector_length_ok) {
        printf("FFT batched RVV: FAIL (invalid VL returned by vsetvl)\n");
        fft_plan_destroy_f32(&plan);
        return 1;
    }
    if (non_finite_count > 0 || max_ratio > 1.0f) {
        printf("FFT batched RVV: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        fft_plan_destroy_f32(&plan);
        return 1;
    }
    printf("FFT batched RVV: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    fft_plan_destroy_f32(&plan);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_batched_testbench_run();
}
#endif
