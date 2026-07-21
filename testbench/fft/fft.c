/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * RVV matrix-form DFT testbench.
 *
 * Dataflow:
 *   output bins are vectorized.
 *   For each complex input scalar x[n], update all output bins with
 *   scalar-vector fused multiply-accumulate operations.
 *
 *   X[k] = sum_n x[n] * exp(-j * 2*pi*k*n/N)
 *
 * This implementation computes a direct DFT using a precomputed
 * twiddle matrix. It is not a radix-2 butterfly FFT.
 */

#define _DEFAULT_SOURCE

#include <math.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_vectors.h"

#define FFT_ATOL 5.0e-5f
#define FFT_RTOL 1.0e-6f

/*
 * Matrix layout:
 *
 * twiddle_real[n][k]
 * twiddle_imag[n][k]
 *
 * For a fixed input n, coefficients of consecutive output bins k
 * are contiguous in memory.
 *
 * This matches the output-channel vectorization used by the
 * im2col Conv1D GEMV kernel.
 */
static float twiddle_real[FFT_SIZE][FFT_SIZE];
static float twiddle_imag[FFT_SIZE][FFT_SIZE];

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

/*
 * Generate the N x N DFT coefficient matrix.
 *
 * W[n][k] = exp(-j * 2*pi*k*n/N)
 *
 * Matrix storage is input-major:
 *
 *     W[n][0], W[n][1], ..., W[n][N-1]
 *
 * Therefore one scalar input x[n] can update multiple output bins
 * through contiguous vector loads.
 */
static void generate_dft_matrix(size_t size)
{
    const float scale = -2.0f * (float)M_PI / (float)size;

    for (size_t input_index = 0; input_index < size; ++input_index) {
        for (size_t bin = 0; bin < size; ++bin) {
            float angle =
                scale * (float)input_index * (float)bin;

            twiddle_real[input_index][bin] = cosf(angle);
            twiddle_imag[input_index][bin] = sinf(angle);
        }
    }
}

/*
 * Compute one output-bin tile.
 *
 * Vector lanes correspond to consecutive output bins:
 *
 *     lane 0 -> X[bin_offset + 0]
 *     lane 1 -> X[bin_offset + 1]
 *     ...
 *
 * Complex MAC:
 *
 *   acc_real += xr * wr - xi * wi
 *   acc_imag += xr * wi + xi * wr
 *
 * The subtraction is expressed as:
 *
 *   acc_real += (-xi) * wi
 *
 * so the implementation only depends on vfmacc.vf semantics.
 */
static void dft_bin_tile_rvv(const float *input_real,
                             const float *input_imag,
                             float *output_real,
                             float *output_imag,
                             size_t size,
                             size_t bin_offset,
                             size_t vl)
{
    vfloat32m4_t acc_real =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);

    vfloat32m4_t acc_imag =
        __riscv_vfmv_v_f_f32m4(0.0f, vl);

    for (size_t input_index = 0;
         input_index < size;
         ++input_index) {
        const float xr = input_real[input_index];
        const float xi = input_imag[input_index];

        const vfloat32m4_t wr =
            __riscv_vle32_v_f32m4(
                &twiddle_real[input_index][bin_offset],
                vl);

        const vfloat32m4_t wi =
            __riscv_vle32_v_f32m4(
                &twiddle_imag[input_index][bin_offset],
                vl);

        /*
         * Real part:
         *
         * acc_real += xr * wr
         * acc_real -= xi * wi
         */
        acc_real =
            __riscv_vfmacc_vf_f32m4(
                acc_real,
                xr,
                wr,
                vl);

        acc_real =
            __riscv_vfmacc_vf_f32m4(
                acc_real,
                -xi,
                wi,
                vl);

        /*
         * Imaginary part:
         *
         * acc_imag += xr * wi
         * acc_imag += xi * wr
         */
        acc_imag =
            __riscv_vfmacc_vf_f32m4(
                acc_imag,
                xr,
                wi,
                vl);

        acc_imag =
            __riscv_vfmacc_vf_f32m4(
                acc_imag,
                xi,
                wr,
                vl);
    }

    __riscv_vse32_v_f32m4(
        &output_real[bin_offset],
        acc_real,
        vl);

    __riscv_vse32_v_f32m4(
        &output_imag[bin_offset],
        acc_imag,
        vl);
}

/*
 * Direct complex DFT using output-bin vectorization.
 *
 * The vector accumulators remain live across the full input dimension.
 * They are stored only after all input samples have been accumulated.
 */
static void dft_rvv_f32(const float *input_real,
                        const float *input_imag,
                        float *output_real,
                        float *output_imag,
                        size_t size)
{
    size_t bin_offset = 0;

    while (bin_offset < size) {
        size_t vl =
            __riscv_vsetvl_e32m4(size - bin_offset);

        dft_bin_tile_rvv(
            input_real,
            input_imag,
            output_real,
            output_imag,
            size,
            bin_offset,
            vl);

        bin_offset += vl;
    }
}

int fft_testbench_run(void)
{
    float actual_real[FFT_SIZE];
    float actual_imag[FFT_SIZE];

    /*
     * Measure twiddle generation separately from kernel execution.
     *
     * This treats the DFT matrix like packed Conv1D weights:
     * coefficients are prepared before inference/execution.
     */
    uint64_t generate_start = read_cycles();
    generate_dft_matrix(FFT_SIZE);
    uint64_t generate_end = read_cycles();

    uint64_t compute_start = read_cycles();
    dft_rvv_f32(
        input_real,
        input_imag,
        actual_real,
        actual_imag,
        FFT_SIZE);
    uint64_t compute_end = read_cycles();

    uint64_t total_cycles = (generate_end - generate_start) + (compute_end - compute_start);

    float max_error = 0.0f;
    size_t max_error_bin = 0;

    float max_error_ratio = 0.0f;
    size_t max_error_ratio_bin = 0;

    for (size_t index = 0;
         index < FFT_SIZE;
         ++index) {
        float real_error =
            fabsf(actual_real[index] -
                  groundtruth_real[index]);

        float imag_error =
            fabsf(actual_imag[index] -
                  groundtruth_imag[index]);

        float error =
            real_error > imag_error
                ? real_error
                : imag_error;

        float real_limit =
            FFT_ATOL +
            FFT_RTOL *
                fabsf(groundtruth_real[index]);

        float imag_limit =
            FFT_ATOL +
            FFT_RTOL *
                fabsf(groundtruth_imag[index]);

        float real_ratio =
            real_error / real_limit;

        float imag_ratio =
            imag_error / imag_limit;

        float error_ratio =
            real_ratio > imag_ratio
                ? real_ratio
                : imag_ratio;

        if (error > max_error) {
            max_error = error;
            max_error_bin = index;
        }

        if (error_ratio > max_error_ratio) {
            max_error_ratio = error_ratio;
            max_error_ratio_bin = index;
        }

        printf(
            "bin %u: %.6f ",
            (unsigned)index,
            actual_real[index]);

        if (actual_imag[index] >= 0.0f) {
            printf("+");
        }

        printf(
            "%.6fj\n",
            actual_imag[index]);
    }

    printf("generate cycles: ");
    print_u64_decimal(generate_end - generate_start);
    printf("\n");

    printf("compute cycles: ");
    print_u64_decimal(compute_end - compute_start);
    printf("\n");

    printf("total cycles: ");
    print_u64_decimal(total_cycles);
    printf("\n");

    printf(
        "max component error: %.6f at bin %u\n",
        max_error,
        (unsigned)max_error_bin);

    printf(
        "max tolerance ratio: %.6f at bin %u\n",
        max_error_ratio,
        (unsigned)max_error_ratio_bin);

    if (max_error_ratio > 1.0f) {
        printf(
            "RVV matrix DFT: FAIL "
            "(atol %.6f, rtol %.6f)\n",
            FFT_ATOL,
            FFT_RTOL);

        return 1;
    }

    printf(
        "RVV matrix DFT: PASS "
        "(atol %.6f, rtol %.6f)\n",
        FFT_ATOL,
        FFT_RTOL);

    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_testbench_run();
}
#endif