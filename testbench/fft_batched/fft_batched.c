/* SPDX-License-Identifier: Apache-2.0 */
/* Batched RVV radix-2 FFT with vectors spanning the batch dimension. */

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"

#define FFT_BATCH_COUNT 64
#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_PI 3.14159265358979323846f

static float input_batch_real[FFT_BATCH_COUNT][FFT_SIZE];
static float input_batch_imag[FFT_BATCH_COUNT][FFT_SIZE];
static float actual_real[FFT_SIZE][FFT_BATCH_COUNT];
static float actual_imag[FFT_SIZE][FFT_BATCH_COUNT];

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

static void fft_batched_rvv_f32(void)
{
    unsigned bits = 0;
    for (size_t value = FFT_SIZE; value > 1; value >>= 1) ++bits;

    for (size_t index = 0; index < FFT_SIZE; ++index) {
        size_t reversed = reverse_bits((unsigned)index, bits);
        if (reversed <= index) continue;
        for (size_t batch = 0; batch < FFT_BATCH_COUNT; ) {
            size_t vl = __riscv_vsetvl_e32m4(FFT_BATCH_COUNT - batch);
            vfloat32m4_t ra = __riscv_vle32_v_f32m4(&actual_real[index][batch], vl);
            vfloat32m4_t ia = __riscv_vle32_v_f32m4(&actual_imag[index][batch], vl);
            vfloat32m4_t rb = __riscv_vle32_v_f32m4(&actual_real[reversed][batch], vl);
            vfloat32m4_t ib = __riscv_vle32_v_f32m4(&actual_imag[reversed][batch], vl);
            __riscv_vse32_v_f32m4(&actual_real[index][batch], rb, vl);
            __riscv_vse32_v_f32m4(&actual_imag[index][batch], ib, vl);
            __riscv_vse32_v_f32m4(&actual_real[reversed][batch], ra, vl);
            __riscv_vse32_v_f32m4(&actual_imag[reversed][batch], ia, vl);
            batch += vl;
        }
    }

    for (size_t block_size = 2; block_size <= FFT_SIZE; block_size <<= 1) {
        size_t half = block_size >> 1;
        for (size_t block = 0; block < FFT_SIZE; block += block_size) {
            for (size_t offset = 0; offset < half; ++offset) {
                float angle = -2.0f * FFT_PI * (float)offset / (float)block_size;
                float wr = cosf(angle), wi = sinf(angle);
                size_t even = block + offset, odd = even + half;
                for (size_t batch = 0; batch < FFT_BATCH_COUNT; ) {
                    size_t vl = __riscv_vsetvl_e32m4(FFT_BATCH_COUNT - batch);
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
}

int main(void)
{
    for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch)
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            input_batch_real[batch][index] = input_real[index];
            input_batch_imag[batch][index] = input_imag[index];
        }

    uint64_t reorder_start = read_cycles();
    reorder_input_to_matrix();
    uint64_t reorder_end = read_cycles();
    uint64_t fft_start = read_cycles();
    fft_batched_rvv_f32();
    uint64_t fft_end = read_cycles();
    uint64_t reorder_cycles = reorder_end - reorder_start;
    uint64_t fft_cycles = fft_end - fft_start;
    float max_error = 0.0f, max_ratio = 0.0f;
    size_t max_bin = 0, max_batch = 0, ratio_bin = 0, ratio_batch = 0;

    for (size_t index = 0; index < FFT_SIZE; ++index) {
        for (size_t batch = 0; batch < FFT_BATCH_COUNT; ++batch) {
            float re = fabsf(actual_real[index][batch] - groundtruth_real[index]);
            float ie = fabsf(actual_imag[index][batch] - groundtruth_imag[index]);
            float error = re > ie ? re : ie;
            float rr = re / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[index]));
            float ir = ie / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[index]));
            float ratio = rr > ir ? rr : ir;
            if (error > max_error) { max_error = error; max_bin = index; max_batch = batch; }
            if (ratio > max_ratio) { max_ratio = ratio; ratio_bin = index; ratio_batch = batch; }
        }
        printf("bin %u: %.6f ", (unsigned)index, actual_real[index][0]);
        if (actual_imag[index][0] >= 0.0f) printf("+");
        printf("%.6fj\n", actual_imag[index][0]);
    }

    printf("batches: %u\nmemory reorder cycles: ", FFT_BATCH_COUNT);
    print_u64_decimal(reorder_cycles);
    printf("\nFFT cycles: "); print_u64_decimal(fft_cycles);
    printf("\ntotal cycles: "); print_u64_decimal(reorder_cycles + fft_cycles);
    printf("\nFFT cycles per FFT: "); print_u64_decimal(fft_cycles / FFT_BATCH_COUNT);
    printf("\ntotal cycles per FFT: "); print_u64_decimal((reorder_cycles + fft_cycles) / FFT_BATCH_COUNT);
    printf("\nmax component error: %.6f at batch %u bin %u\n", max_error, (unsigned)max_batch, (unsigned)max_bin);
    printf("max tolerance ratio: %.6f at batch %u bin %u\n", max_ratio, (unsigned)ratio_batch, (unsigned)ratio_bin);
    if (max_ratio > 1.0f) {
        printf("FFT batched RVV: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        return 1;
    }
    printf("FFT batched RVV: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    return 0;
}
