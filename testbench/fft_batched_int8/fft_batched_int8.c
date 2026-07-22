/* SPDX-License-Identifier: Apache-2.0 */
/* Batched radix-2 scaled Q7 FFT with vectors spanning the batch dimension. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "fft_batched_int8_runner.h"
#include "fft_int8_vectors.h"
#include "nn_runtime.h"

#define FFT_INT8_BATCH_COUNT 64
#define FFT_INT8_MAX_MISMATCH_REPORTS 64

typedef struct {
    size_t size;
    size_t batch_count;
    unsigned stage_count;
    int8_t *data_real;
    int8_t *data_imag;
    int8_t *twiddle_real;
    int8_t *twiddle_imag;
} fft_plan_q7_t;

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

static inline size_t data_offset(const fft_plan_q7_t *plan,
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

static int8_t quantize_q7(float value)
{
    long scaled = lroundf(value * 128.0f);
    if (scaled > 127) scaled = 127;
    if (scaled < -128) scaled = -128;
    return (int8_t)scaled;
}

static int fft_plan_init_q7(fft_plan_q7_t *plan,
                            size_t size,
                            size_t batch_count)
{
    if (size < 2 || size > FFT_INT8_SIZE || (size & (size - 1)) != 0 ||
        batch_count == 0 || batch_count > FFT_INT8_BATCH_COUNT) return 0;

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

static void fft_plan_destroy_q7(fft_plan_q7_t *plan)
{
    safe_free(plan->twiddle_imag);
    safe_free(plan->twiddle_real);
    safe_free(plan->data_imag);
    safe_free(plan->data_real);
}

static void fft_plan_load_input_q7(const fft_plan_q7_t *plan)
{
    for (size_t index = 0; index < plan->size; ++index) {
        size_t row = index * plan->batch_count;
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            plan->data_real[row + batch] = fft_int8_input_real[index];
            plan->data_imag[row + batch] = fft_int8_input_imag[index];
        }
    }
}

static void swap_rows_e8m1(int8_t *a_real,
                           int8_t *a_imag,
                           int8_t *b_real,
                           int8_t *b_imag,
                           size_t count)
{
    while (count != 0) {
        size_t vl = __riscv_vsetvl_e8m1(count);
        vint8m1_t ar = __riscv_vle8_v_i8m1(a_real, vl);
        vint8m1_t ai = __riscv_vle8_v_i8m1(a_imag, vl);
        vint8m1_t br = __riscv_vle8_v_i8m1(b_real, vl);
        vint8m1_t bi = __riscv_vle8_v_i8m1(b_imag, vl);
        __riscv_vse8_v_i8m1(a_real, br, vl);
        __riscv_vse8_v_i8m1(a_imag, bi, vl);
        __riscv_vse8_v_i8m1(b_real, ar, vl);
        __riscv_vse8_v_i8m1(b_imag, ai, vl);
        a_real += vl; a_imag += vl; b_real += vl; b_imag += vl; count -= vl;
    }
}

static inline void butterfly_q7(int8_t *restrict even_real,
                                int8_t *restrict even_imag,
                                int8_t *restrict odd_real,
                                int8_t *restrict odd_imag,
                                int8_t wr,
                                int8_t wi,
                                size_t vl)
{
    vint8m2_t or_ = __riscv_vle8_v_i8m2(odd_real, vl);
    vint8m2_t oi = __riscv_vle8_v_i8m2(odd_imag, vl);

    vint16m4_t product16 = __riscv_vwmul_vx_i16m4(or_, wr, vl);
    vint16m4_t other16 = __riscv_vwmul_vx_i16m4(oi, wi, vl);
    vint32m8_t product = __riscv_vsext_vf2_i32m8(product16, vl);
    vint32m8_t other = __riscv_vsext_vf2_i32m8(other16, vl);
    product = __riscv_vsub_vv_i32m8(product, other, vl);
    product16 = __riscv_vnclip_wx_i16m4(
        product, 7, __RISCV_VXRM_RNU, vl);
    vint8m2_t tr = __riscv_vnclip_wx_i8m2(
        product16, 0, __RISCV_VXRM_RNU, vl);

    product16 = __riscv_vwmul_vx_i16m4(oi, wr, vl);
    other16 = __riscv_vwmul_vx_i16m4(or_, wi, vl);
    product = __riscv_vsext_vf2_i32m8(product16, vl);
    other = __riscv_vsext_vf2_i32m8(other16, vl);
    product = __riscv_vadd_vv_i32m8(product, other, vl);
    product16 = __riscv_vnclip_wx_i16m4(
        product, 7, __RISCV_VXRM_RNU, vl);
    vint8m2_t ti = __riscv_vnclip_wx_i8m2(
        product16, 0, __RISCV_VXRM_RNU, vl);

    vint8m2_t er = __riscv_vle8_v_i8m2(even_real, vl);
    vint8m2_t upper_r = __riscv_vaadd_vv_i8m2(
        er, tr, __RISCV_VXRM_RNU, vl);
    vint8m2_t lower_r = __riscv_vasub_vv_i8m2(
        er, tr, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(even_real, upper_r, vl);
    __riscv_vse8_v_i8m2(odd_real, lower_r, vl);

    vint8m2_t ei = __riscv_vle8_v_i8m2(even_imag, vl);
    vint8m2_t upper_i = __riscv_vaadd_vv_i8m2(
        ei, ti, __RISCV_VXRM_RNU, vl);
    vint8m2_t lower_i = __riscv_vasub_vv_i8m2(
        ei, ti, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(even_imag, upper_i, vl);
    __riscv_vse8_v_i8m2(odd_imag, lower_i, vl);
}

static inline void butterfly_unity_q7(int8_t *restrict even_real,
                                      int8_t *restrict even_imag,
                                      int8_t *restrict odd_real,
                                      int8_t *restrict odd_imag,
                                      size_t vl)
{
    vint8m2_t or_ = __riscv_vle8_v_i8m2(odd_real, vl);
    vint8m2_t er = __riscv_vle8_v_i8m2(even_real, vl);
    vint8m2_t upper_r = __riscv_vaadd_vv_i8m2(er, or_, __RISCV_VXRM_RNU, vl);
    vint8m2_t lower_r = __riscv_vasub_vv_i8m2(er, or_, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(even_real, upper_r, vl);
    __riscv_vse8_v_i8m2(odd_real, lower_r, vl);

    vint8m2_t oi = __riscv_vle8_v_i8m2(odd_imag, vl);
    vint8m2_t ei = __riscv_vle8_v_i8m2(even_imag, vl);
    vint8m2_t upper_i = __riscv_vaadd_vv_i8m2(ei, oi, __RISCV_VXRM_RNU, vl);
    vint8m2_t lower_i = __riscv_vasub_vv_i8m2(ei, oi, __RISCV_VXRM_RNU, vl);
    __riscv_vse8_v_i8m2(even_imag, upper_i, vl);
    __riscv_vse8_v_i8m2(odd_imag, lower_i, vl);
}

static int fft_batched_rvv_q7(const fft_plan_q7_t *plan)
{
    for (size_t index = 0; index < plan->size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, plan->stage_count);
        if (reversed <= index) continue;
        size_t a = data_offset(plan, index, 0);
        size_t b = data_offset(plan, reversed, 0);
        swap_rows_e8m1(&plan->data_real[a], &plan->data_imag[a],
                       &plan->data_real[b], &plan->data_imag[b],
                       plan->batch_count);
    }

    size_t vlmax = __riscv_vsetvlmax_e8m2();
    if (vlmax == 0) return 0;
    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;
        for (size_t offset = 0; offset < half; ++offset) {
            int8_t wr = plan->twiddle_real[stage_base + offset];
            int8_t wi = plan->twiddle_imag[stage_base + offset];
            for (size_t block = 0; block < plan->size; block += block_size) {
                size_t even_base = data_offset(plan, block + offset, 0);
                size_t odd_base = data_offset(plan, block + offset + half, 0);
                size_t batch = 0;
                for (; plan->batch_count - batch >= vlmax; batch += vlmax) {
                    if (offset == 0) {
                        butterfly_unity_q7(&plan->data_real[even_base + batch],
                            &plan->data_imag[even_base + batch],
                            &plan->data_real[odd_base + batch],
                            &plan->data_imag[odd_base + batch], vlmax);
                    } else {
                        butterfly_q7(&plan->data_real[even_base + batch],
                            &plan->data_imag[even_base + batch],
                            &plan->data_real[odd_base + batch],
                            &plan->data_imag[odd_base + batch], wr, wi, vlmax);
                    }
                }
                if (batch < plan->batch_count) {
                    size_t vl = __riscv_vsetvl_e8m2(plan->batch_count - batch);
                    if (offset == 0) {
                        butterfly_unity_q7(&plan->data_real[even_base + batch],
                            &plan->data_imag[even_base + batch],
                            &plan->data_real[odd_base + batch],
                            &plan->data_imag[odd_base + batch], vl);
                    } else {
                        butterfly_q7(&plan->data_real[even_base + batch],
                            &plan->data_imag[even_base + batch],
                            &plan->data_real[odd_base + batch],
                            &plan->data_imag[odd_base + batch], wr, wi, vl);
                    }
                }
            }
        }
    }
    return 1;
}

int fft_batched_int8_testbench_run(void)
{
    fft_plan_q7_t plan;
    uint64_t plan_start = nn_runtime_read_cycles();
    int plan_ok = fft_plan_init_q7(&plan, FFT_INT8_SIZE, FFT_INT8_BATCH_COUNT);
    uint64_t plan_end = nn_runtime_read_cycles();
    if (!plan_ok) {
        printf("FFT batched int8 RVV: FAIL (invalid plan)\n");
        return 1;
    }
    uint64_t layout_start = nn_runtime_read_cycles();
    fft_plan_load_input_q7(&plan);
    uint64_t layout_end = nn_runtime_read_cycles();
    uint64_t fft_start = nn_runtime_read_cycles();
    int vl_ok = fft_batched_rvv_q7(&plan);
    uint64_t fft_end = nn_runtime_read_cycles();

    size_t mismatches = 0, reported = 0;
    int max_error = 0;
    size_t max_batch = 0, max_bin = 0;
    for (size_t bin = 0; bin < plan.size; ++bin) {
        for (size_t batch = 0; batch < plan.batch_count; ++batch) {
            size_t pos = data_offset(&plan, bin, batch);
            int dr = (int)plan.data_real[pos] - (int)fft_int8_groundtruth_real[bin];
            int di = (int)plan.data_imag[pos] - (int)fft_int8_groundtruth_imag[bin];
            int error = dr < 0 ? -dr : dr;
            int imag_error = di < 0 ? -di : di;
            if (imag_error > error) error = imag_error;
            if (error > max_error) { max_error = error; max_batch = batch; max_bin = bin; }
            if (dr != 0 || di != 0) {
                ++mismatches;
                if (reported < FFT_INT8_MAX_MISMATCH_REPORTS) {
                    printf("mismatch batch %u bin %u: actual %d %dj expected %d %dj\n",
                        (unsigned)batch, (unsigned)bin,
                        (int)plan.data_real[pos], (int)plan.data_imag[pos],
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

    if (!vl_ok || mismatches != 0) {
        printf("FFT batched int8 RVV: FAIL (bit-exact Q7 validation)\n");
        fft_plan_destroy_q7(&plan);
        return 1;
    }
    printf("FFT batched int8 RVV: PASS (bit-exact Q7 validation)\n");
    fft_plan_destroy_q7(&plan);
    return 0;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_batched_int8_testbench_run();
}
#endif
