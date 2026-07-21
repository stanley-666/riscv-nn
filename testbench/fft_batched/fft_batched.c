/* SPDX-License-Identifier: Apache-2.0 */
/* Batched RVV radix-2 FFT with vectors spanning the batch dimension. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <riscv_vector.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"

#define FFT_BATCH_COUNT 64
#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_MAX_MISMATCH_REPORTS 64

static float input_batch_real[FFT_BATCH_COUNT][FFT_SIZE]
    __attribute__((aligned(64)));
static float input_batch_imag[FFT_BATCH_COUNT][FFT_SIZE]
    __attribute__((aligned(64)));
static float actual_real[FFT_SIZE][FFT_BATCH_COUNT]
    __attribute__((aligned(64)));
static float actual_imag[FFT_SIZE][FFT_BATCH_COUNT]
    __attribute__((aligned(64)));

static uint64_t read_cycles(void)
{
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles) :: "memory");
    return cycles;
}

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

static void reorder_input_to_matrix(void)
{
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch)
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            actual_real[index][batch] = input_batch_real[batch][index];
            actual_imag[index][batch] = input_batch_imag[batch][index];
        }
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

static int fft_batched_rvv_f32(void)
{
    unsigned bits = 0;
    for (size_t value = FFT_SIZE; value > 1; value >>= 1) ++bits;

    for (size_t index = 0; index < FFT_SIZE; ++index) {
        size_t reversed = reverse_bits((unsigned)index, bits);
        if (reversed <= index) continue;
        for (size_t batch = 0; batch < FFT_BATCH_COUNT; ) {
            size_t remaining = FFT_BATCH_COUNT - batch;
            size_t vl;
            __asm__ volatile(
                "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
                "vle32.v v0, (%[ra])\n\t"
                "vle32.v v1, (%[ia])\n\t"
                "vle32.v v2, (%[rb])\n\t"
                "vle32.v v3, (%[ib])\n\t"
                "vse32.v v2, (%[ra])\n\t"
                "vse32.v v3, (%[ia])\n\t"
                "vse32.v v0, (%[rb])\n\t"
                "vse32.v v1, (%[ib])"
                : [vl] "=&r"(vl)
                : [avl] "r"(remaining),
                  [ra] "r"(&actual_real[index][batch]),
                  [ia] "r"(&actual_imag[index][batch]),
                  [rb] "r"(&actual_real[reversed][batch]),
                  [ib] "r"(&actual_imag[reversed][batch])
                : "v0", "v1", "v2", "v3", "memory");
            if (vl == 0 || vl > remaining) return 0;
            batch += vl;
        }
    }

    for (size_t block_size = 2; block_size <= FFT_SIZE; block_size <<= 1) {
        size_t half = block_size >> 1;
        for (size_t block = 0; block < FFT_SIZE; block += block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                float angle = -2.0f * (float)M_PI * (float)offset / (float)block_size;
                float wr = cosf(angle), wi = sinf(angle);
                size_t even = block + offset, odd = even + half;
                for (size_t batch = 0; batch < FFT_BATCH_COUNT; ) {
                    size_t remaining = FFT_BATCH_COUNT - batch;
                    size_t vl = __riscv_vsetvl_e32m4(remaining);
                    if (vl == 0 || vl > remaining) return 0;
                    vfloat32m4_t er = __riscv_vle32_v_f32m4(&actual_real[even][batch], vl);
                    vfloat32m4_t ei = __riscv_vle32_v_f32m4(&actual_imag[even][batch], vl);
                    vfloat32m4_t or = __riscv_vle32_v_f32m4(&actual_real[odd][batch], vl);
                    vfloat32m4_t oi = __riscv_vle32_v_f32m4(&actual_imag[odd][batch], vl);
                    vfloat32m4_t pr = __riscv_vfsub_vv_f32m4(
                        __riscv_vfmul_vf_f32m4(or, wr, vl),
                        __riscv_vfmul_vf_f32m4(oi, wi, vl), vl);
                    vfloat32m4_t pi = __riscv_vfadd_vv_f32m4(
                        __riscv_vfmul_vf_f32m4(oi, wr, vl),
                        __riscv_vfmul_vf_f32m4(or, wi, vl), vl);
                    __riscv_vse32_v_f32m4(&actual_real[even][batch], __riscv_vfadd_vv_f32m4(er, pr, vl), vl);
                    __riscv_vse32_v_f32m4(&actual_imag[even][batch], __riscv_vfadd_vv_f32m4(ei, pi, vl), vl);
                    __riscv_vse32_v_f32m4(&actual_real[odd][batch], __riscv_vfsub_vv_f32m4(er, pr, vl), vl);
                    __riscv_vse32_v_f32m4(&actual_imag[odd][batch], __riscv_vfsub_vv_f32m4(ei, pi, vl), vl);
                    batch += vl;
                }
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

    /* Every radix-2 intermediate component is bounded by
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

    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch)
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            input_batch_real[batch][index] = input_real[index];
            input_batch_imag[batch][index] = input_imag[index];
        }

    uint64_t reorder_start = read_cycles();
    reorder_input_to_matrix();
    uint64_t reorder_end = read_cycles();
    uint64_t fft_start = read_cycles();
    int vector_length_ok = fft_batched_rvv_f32();
    uint64_t fft_end = read_cycles();
    uint64_t reorder_cycles = reorder_end - reorder_start;
    uint64_t fft_cycles = fft_end - fft_start;
    float max_error = 0.0f, max_ratio = 0.0f;
    size_t max_bin = 0, max_batch = 0, ratio_bin = 0, ratio_batch = 0;
    size_t non_finite_count = 0, first_non_finite_bin = 0, first_non_finite_batch = 0;
    size_t mismatch_count = 0, reported_mismatch_count = 0;
    size_t batch_mismatch_count[FFT_BATCH_COUNT] = {0};

    for (size_t index = 0; index < FFT_SIZE; ++index) {
        for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch) {
            if (!isfinite(actual_real[index][batch]) ||
                !isfinite(actual_imag[index][batch])) {
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
            float re = fabsf(actual_real[index][batch] - groundtruth_real[index]);
            float ie = fabsf(actual_imag[index][batch] - groundtruth_imag[index]);
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
                    printf("  actual:   %.6f ", actual_real[index][batch]);
                    if (actual_imag[index][batch] >= 0.0f) printf("+");
                    printf("%.6fj\n", actual_imag[index][batch]);
                    printf("  expected: %.6f ", groundtruth_real[index]);
                    if (groundtruth_imag[index] >= 0.0f) printf("+");
                    printf("%.6fj\n", groundtruth_imag[index]);
                    printf("  error:    real %.6f imag %.6f\n", re, ie);
                    ++reported_mismatch_count;
                }
            }
        }
    }

    printf("batches: %u\nmemory reorder cycles: ", FFT_BATCH_COUNT);
    print_u64_decimal(reorder_cycles);
    printf("\nFFT cycles: "); print_u64_decimal(fft_cycles);
    printf("\ntotal cycles: "); print_u64_decimal(reorder_cycles + fft_cycles);
    printf("\nFFT cycles per FFT: "); print_u64_decimal(fft_cycles / FFT_BATCH_COUNT);
    printf("\ntotal cycles per FFT: "); print_u64_decimal((reorder_cycles + fft_cycles) / FFT_BATCH_COUNT);
    printf("\nmax component error: %.6f at batch %u bin %u\n", max_error, (unsigned)max_batch, (unsigned)max_bin);
    printf("max tolerance ratio: %.6f at batch %u bin %u\n", max_ratio, (unsigned)ratio_batch, (unsigned)ratio_bin);
    printf("mismatched complex points: %u / %u\n", (unsigned)mismatch_count,
        (unsigned)(FFT_BATCH_COUNT * FFT_SIZE));
    if (mismatch_count > reported_mismatch_count) {
        printf("mismatch report truncated: showed first %u of %u points\n",
            (unsigned)reported_mismatch_count, (unsigned)mismatch_count);
    }
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch) {
        if (batch_mismatch_count[batch] != 0) {
            printf("batch %u mismatches: %u / %u\n", (unsigned)batch,
                (unsigned)batch_mismatch_count[batch], (unsigned)FFT_SIZE);
        }
    }
    printf("non-finite outputs: %u\n", (unsigned)non_finite_count);
    if (non_finite_count > 0) {
        printf("first non-finite output: batch %u bin %u\n",
            (unsigned)first_non_finite_batch, (unsigned)first_non_finite_bin);
    }
    if (!vector_length_ok) {
        printf("FFT batched RVV: FAIL (invalid VL returned by vsetvl)\n");
        return 1;
    }
    if (non_finite_count > 0 || max_ratio > 1.0f) {
        printf("FFT batched RVV: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        return 1;
    }
    printf("FFT batched RVV: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_batched_testbench_run();
}
#endif
