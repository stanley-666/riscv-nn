/*
 * SPDX-License-Identifier: Apache-2.0
 * RVV radix-2 FFT testbench with a NumPy-compatible ground truth.
 */

#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"

#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f
#define FFT_PI 3.14159265358979323846f

static uint64_t read_cycles(void)
{
    uint64_t cycles;
    __asm__ volatile("rdcycle %0" : "=r"(cycles));
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
    while (count > 0u) {
        printf("%c", digits[--count]);
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

static void fft_rvv_f32(float *real, float *imag, size_t size)
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

    float twiddle_real[FFT_SIZE / 2];
    float twiddle_imag[FFT_SIZE / 2];
    for (size_t block_size = 2; block_size <= size; block_size <<= 1) {
        size_t half = block_size >> 1;
        for (size_t offset = 0; offset < half; ++offset) {
            float angle = -2.0f * FFT_PI * (float)offset / (float)block_size;
            twiddle_real[offset] = cosf(angle);
            twiddle_imag[offset] = sinf(angle);
        }

        for (size_t block = 0; block < size; block += block_size) {
            size_t offset = 0;
            while (offset < half) {
                size_t vl = __riscv_vsetvl_e32m4(half - offset);
                vfloat32m4_t even_real = __riscv_vle32_v_f32m4(&real[block + offset], vl);
                vfloat32m4_t even_imag = __riscv_vle32_v_f32m4(&imag[block + offset], vl);
                vfloat32m4_t odd_real = __riscv_vle32_v_f32m4(&real[block + half + offset], vl);
                vfloat32m4_t odd_imag = __riscv_vle32_v_f32m4(&imag[block + half + offset], vl);
                vfloat32m4_t wr = __riscv_vle32_v_f32m4(&twiddle_real[offset], vl);
                vfloat32m4_t wi = __riscv_vle32_v_f32m4(&twiddle_imag[offset], vl);

                vfloat32m4_t product_real = __riscv_vfsub_vv_f32m4(
                    __riscv_vfmul_vv_f32m4(wr, odd_real, vl),
                    __riscv_vfmul_vv_f32m4(wi, odd_imag, vl), vl);
                vfloat32m4_t product_imag = __riscv_vfadd_vv_f32m4(
                    __riscv_vfmul_vv_f32m4(wr, odd_imag, vl),
                    __riscv_vfmul_vv_f32m4(wi, odd_real, vl), vl);

                __riscv_vse32_v_f32m4(&real[block + offset],
                    __riscv_vfadd_vv_f32m4(even_real, product_real, vl), vl);
                __riscv_vse32_v_f32m4(&imag[block + offset],
                    __riscv_vfadd_vv_f32m4(even_imag, product_imag, vl), vl);
                __riscv_vse32_v_f32m4(&real[block + half + offset],
                    __riscv_vfsub_vv_f32m4(even_real, product_real, vl), vl);
                __riscv_vse32_v_f32m4(&imag[block + half + offset],
                    __riscv_vfsub_vv_f32m4(even_imag, product_imag, vl), vl);
                offset += vl;
            }
        }
    }
}

int fft_testbench_run(void)
{
    float actual_real[FFT_SIZE];
    float actual_imag[FFT_SIZE];
    for (size_t index = 0; index < FFT_SIZE; ++index) {
        actual_real[index] = input_real[index];
        actual_imag[index] = input_imag[index];
    }

    uint64_t start = read_cycles();
    fft_rvv_f32(actual_real, actual_imag, FFT_SIZE);
    uint64_t end = read_cycles();

    float max_error = 0.0f;
    size_t max_error_bin = 0;
    float max_error_ratio = 0.0f;
    size_t max_error_ratio_bin = 0;
    for (size_t index = 0; index < FFT_SIZE; ++index) {
        float real_error = fabsf(actual_real[index] - groundtruth_real[index]);
        float imag_error = fabsf(actual_imag[index] - groundtruth_imag[index]);
        float error = real_error > imag_error ? real_error : imag_error;
        float real_limit = FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[index]);
        float imag_limit = FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[index]);
        float real_ratio = real_error / real_limit;
        float imag_ratio = imag_error / imag_limit;
        float error_ratio = real_ratio > imag_ratio ? real_ratio : imag_ratio;
        if (error > max_error) {
            max_error = error;
            max_error_bin = index;
        }
        if (error_ratio > max_error_ratio) {
            max_error_ratio = error_ratio;
            max_error_ratio_bin = index;
        }
        printf("bin %u: %.6f ", (unsigned)index, actual_real[index]);
        if (actual_imag[index] >= 0.0f) {
            printf("+");
        }
        printf("%.6fj\n", actual_imag[index]);
    }

    printf("cycles: ");
    print_u64_decimal(end - start);
    printf("\n");
    printf("max component error: %.6f at bin %u\n", max_error, (unsigned)max_error_bin);
    printf("max tolerance ratio: %.6f at bin %u\n", max_error_ratio, (unsigned)max_error_ratio_bin);
    if (max_error_ratio > 1.0f) {
        printf("FFT RVV: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        return 1;
    }
    printf("FFT RVV: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_testbench_run();
}
#endif
