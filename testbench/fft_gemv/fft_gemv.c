/* SPDX-License-Identifier: Apache-2.0 */
/* Complex DFT expressed as Conv1D-style scalar-vector GEMV MACs. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"

#define FFT_ATOL 2.0e-4f
#define FFT_RTOL 1.0e-6f

/* Natural DFT layout: [output bin][input sample]. */
static float twiddle_real_original[FFT_SIZE][FFT_SIZE]
    __attribute__((aligned(64)));
static float twiddle_imag_original[FFT_SIZE][FFT_SIZE]
    __attribute__((aligned(64)));

/* Conv1D/GEMV layout: [scalar input][contiguous output weights]. */
static float twiddle_real_rvv[FFT_SIZE][FFT_SIZE]
    __attribute__((aligned(64)));
static float twiddle_imag_rvv[FFT_SIZE][FFT_SIZE]
    __attribute__((aligned(64)));
static float actual_real[FFT_SIZE] __attribute__((aligned(64)));
static float actual_imag[FFT_SIZE] __attribute__((aligned(64)));

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
    while (count > 0u) {
        printf("%c", digits[--count]);
    }
}

static void generate_twiddle_matrix(void)
{
    for (size_t output_bin = 0; output_bin < FFT_SIZE; ++output_bin) {
        for (size_t input_index = 0; input_index < FFT_SIZE; ++input_index) {
            double angle = -2.0 * M_PI * (double)output_bin
                * (double)input_index / (double)FFT_SIZE;
            twiddle_real_original[output_bin][input_index] = (float)cos(angle);
            twiddle_imag_original[output_bin][input_index] = (float)sin(angle);
        }
    }
}

static void reorder_twiddle_for_scalar_vector_mac(void)
{
    for (size_t output_bin = 0; output_bin < FFT_SIZE; ++output_bin) {
        for (size_t input_index = 0; input_index < FFT_SIZE; ++input_index) {
            twiddle_real_rvv[input_index][output_bin] =
                twiddle_real_original[output_bin][input_index];
            twiddle_imag_rvv[input_index][output_bin] =
                twiddle_imag_original[output_bin][input_index];
        }
    }
}

static void dft_gemv_rvv_f32(void)
{
    size_t output_bin = 0;
    while (output_bin < FFT_SIZE) {
        size_t vl = __riscv_vsetvl_e32m8(FFT_SIZE - output_bin);
        vfloat32m8_t output_real = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t output_imag = __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (size_t input_index = 0; input_index < FFT_SIZE; ++input_index) {
            float input_r = input_real[input_index];
            float input_i = input_imag[input_index];
            vfloat32m8_t weight_real = __riscv_vle32_v_f32m8(
                &twiddle_real_rvv[input_index][output_bin], vl);
            vfloat32m8_t weight_imag = __riscv_vle32_v_f32m8(
                &twiddle_imag_rvv[input_index][output_bin], vl);

            output_real = __riscv_vfmacc_vf_f32m8(
                output_real, input_r, weight_real, vl);
            output_real = __riscv_vfnmsac_vf_f32m8(
                output_real, input_i, weight_imag, vl);
            output_imag = __riscv_vfmacc_vf_f32m8(
                output_imag, input_i, weight_real, vl);
            output_imag = __riscv_vfmacc_vf_f32m8(
                output_imag, input_r, weight_imag, vl);
        }

        __riscv_vse32_v_f32m8(&actual_real[output_bin], output_real, vl);
        __riscv_vse32_v_f32m8(&actual_imag[output_bin], output_imag, vl);
        output_bin += vl;
    }
}

int fft_gemv_testbench_run(void)
{
    uint64_t generation_start = read_cycles();
    generate_twiddle_matrix();
    uint64_t generation_end = read_cycles();
    uint64_t reorder_start = read_cycles();
    reorder_twiddle_for_scalar_vector_mac();
    uint64_t reorder_end = read_cycles();
    uint64_t gemv_start = read_cycles();
    dft_gemv_rvv_f32();
    uint64_t gemv_end = read_cycles();

    uint64_t generation_cycles = generation_end - generation_start;
    uint64_t reorder_cycles = reorder_end - reorder_start;
    uint64_t gemv_cycles = gemv_end - gemv_start;
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

    printf("twiddle generation cycles: ");
    print_u64_decimal(generation_cycles);
    printf("\nmemory reorder cycles: ");
    print_u64_decimal(reorder_cycles);
    printf("\nGEMV cycles: ");
    print_u64_decimal(gemv_cycles);
    printf("\ntotal cycles: ");
    print_u64_decimal(generation_cycles + reorder_cycles + gemv_cycles);
    printf("\nmax component error: %.6f at bin %u\n", max_error,
        (unsigned)max_error_bin);
    printf("max tolerance ratio: %.6f at bin %u\n", max_error_ratio,
        (unsigned)max_error_ratio_bin);
    if (max_error_ratio > 1.0f) {
        printf("DFT GEMV RVV: FAIL (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
        return 1;
    }
    printf("DFT GEMV RVV: PASS (atol %.6f, rtol %.6f)\n", FFT_ATOL, FFT_RTOL);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_gemv_testbench_run();
}
#endif
