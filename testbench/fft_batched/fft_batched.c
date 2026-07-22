/* SPDX-License-Identifier: Apache-2.0 */
/* Batched radix-2 FFT with vectors spanning the batch dimension. */

#define _DEFAULT_SOURCE

#include <float.h>
#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdatomic.h>
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
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch) {
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            actual_real[index][batch] = input_batch_real[batch][index];
            actual_imag[index][batch] = input_batch_imag[batch][index];
        }
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

static inline void twiddle_f32(size_t offset,
                               size_t block_size,
                               float *twiddle_real,
                               float *twiddle_imag)
{
    float angle = -2.0f * (float)M_PI * (float)offset /
        (float)block_size;
    *twiddle_real = cosf(angle);
    *twiddle_imag = sinf(angle);
}

static inline void butterfly_rvv_f32(float *even_real,
                                     float *even_imag,
                                     float *odd_real,
                                     float *odd_imag,
                                     float twiddle_real,
                                     float twiddle_imag,
                                     size_t vl)
{
    vfloat32m8_t odd_r = __riscv_vle32_v_f32m8(odd_real, vl);
    vfloat32m8_t product_r =
        __riscv_vfmul_vf_f32m8(odd_r, twiddle_real, vl);
    atomic_signal_fence(memory_order_seq_cst);

    vfloat32m8_t odd_i = __riscv_vle32_v_f32m8(odd_imag, vl);
    product_r = __riscv_vfnmsac_vf_f32m8(
        product_r, twiddle_imag, odd_i, vl);

    vfloat32m8_t product_i =
        __riscv_vfmul_vf_f32m8(odd_i, twiddle_real, vl);
    product_i = __riscv_vfmacc_vf_f32m8(
        product_i, twiddle_imag, odd_r, vl);
    atomic_signal_fence(memory_order_seq_cst);

    vfloat32m8_t even_r = __riscv_vle32_v_f32m8(even_real, vl);
    __riscv_vse32_v_f32m8(even_real,
        __riscv_vfadd_vv_f32m8(even_r, product_r, vl), vl);
    __riscv_vse32_v_f32m8(odd_real,
        __riscv_vfsub_vv_f32m8(even_r, product_r, vl), vl);

    vfloat32m8_t even_i = __riscv_vle32_v_f32m8(even_imag, vl);
    __riscv_vse32_v_f32m8(even_imag,
        __riscv_vfadd_vv_f32m8(even_i, product_i, vl), vl);
    __riscv_vse32_v_f32m8(odd_imag,
        __riscv_vfsub_vv_f32m8(even_i, product_i, vl), vl);
}

static inline void butterfly_unity_rvv_f32(float *even_real,
                                           float *even_imag,
                                           float *odd_real,
                                           float *odd_imag,
                                           size_t vl)
{
    vfloat32m8_t odd_r = __riscv_vle32_v_f32m8(odd_real, vl);
    vfloat32m8_t even_r = __riscv_vle32_v_f32m8(even_real, vl);
    __riscv_vse32_v_f32m8(even_real,
        __riscv_vfadd_vv_f32m8(even_r, odd_r, vl), vl);
    __riscv_vse32_v_f32m8(odd_real,
        __riscv_vfsub_vv_f32m8(even_r, odd_r, vl), vl);

    vfloat32m8_t odd_i = __riscv_vle32_v_f32m8(odd_imag, vl);
    vfloat32m8_t even_i = __riscv_vle32_v_f32m8(even_imag, vl);
    __riscv_vse32_v_f32m8(even_imag,
        __riscv_vfadd_vv_f32m8(even_i, odd_i, vl), vl);
    __riscv_vse32_v_f32m8(odd_imag,
        __riscv_vfsub_vv_f32m8(even_i, odd_i, vl), vl);
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
            size_t vl = __riscv_vsetvl_e32m1(remaining);
            if (vl == 0 || vl > remaining) return 0;
            vfloat32m1_t ar = __riscv_vle32_v_f32m1(&actual_real[index][batch], vl);
            vfloat32m1_t ai = __riscv_vle32_v_f32m1(&actual_imag[index][batch], vl);
            vfloat32m1_t br = __riscv_vle32_v_f32m1(&actual_real[reversed][batch], vl);
            vfloat32m1_t bi = __riscv_vle32_v_f32m1(&actual_imag[reversed][batch], vl);
            __riscv_vse32_v_f32m1(&actual_real[index][batch], br, vl);
            __riscv_vse32_v_f32m1(&actual_imag[index][batch], bi, vl);
            __riscv_vse32_v_f32m1(&actual_real[reversed][batch], ar, vl);
            __riscv_vse32_v_f32m1(&actual_imag[reversed][batch], ai, vl);
            batch += vl;
        }
    }

    for (size_t block_size = 2; block_size <= FFT_SIZE; block_size <<= 1) {
        size_t half = block_size >> 1;
        for (size_t offset = 0; offset < half; ++offset) {
            float wr = 1.0f;
            float wi = 0.0f;
            if (offset != 0) {
                twiddle_f32(offset, block_size, &wr, &wi);
            }

            for (size_t block = 0; block < FFT_SIZE;
                 block += block_size) {
                size_t even = block + offset;
                size_t odd = even + half;

                for (size_t batch = 0; batch < FFT_BATCH_COUNT; ) {
                    size_t remaining = FFT_BATCH_COUNT - batch;
                    size_t vl = __riscv_vsetvl_e32m8(remaining);
                    if (vl == 0 || vl > remaining) return 0;
                    if (offset == 0) {
                        butterfly_unity_rvv_f32(
                            &actual_real[even][batch],
                            &actual_imag[even][batch],
                            &actual_real[odd][batch],
                            &actual_imag[odd][batch], vl);
                    } else {
                        butterfly_rvv_f32(
                            &actual_real[even][batch],
                            &actual_imag[even][batch],
                            &actual_real[odd][batch],
                            &actual_imag[odd][batch], wr, wi, vl);
                    }
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
