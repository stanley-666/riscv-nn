/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
/*
 * FP32 batched radix-2 FFT layout experiment.
 *
 * This file contains floating-point kernels only:
 *   - bin-major [bin][batch], with RVV lanes spanning batches;
 *   - batch-major [batch][bin], with RVV lanes spanning FFT bins.
 *
 * The separate testbench/fft_batched_int8/ directory contains the Q7/int8
 * implementation.
 */

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
#define FFT_BENCHMARK_RUNS 10u

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
    /*
     * BIN-MAJOR layout: [bin][batch]
     * linear offset = bin * batch_count + batch.
     * Vector lanes span independent FFT batches, allowing one scalar twiddle
     * to be broadcast across the complete batch vector.
     */
    for (size_t index = 0; index < plan->size; ++index) {
        size_t row = index * plan->batch_count;
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            plan->data_real[row + batch] = input_real[index];
            plan->data_imag[row + batch] = input_imag[index];
        }
    }
    return 1;
}

static int fft_plan_load_input_batch_major_f32(const fft_plan_f32_t *plan)
{
    /*
     * BATCH-MAJOR layout: [batch][bin]
     * linear offset = batch * FFT_size + bin.
     * Vector lanes span consecutive bins of one FFT; twiddles must therefore
     * be loaded as vectors instead of using one scalar broadcast.
     */
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t bin = 0; bin < plan->size; ++bin) {
            plan->data_real[base + bin] = input_real[bin];
            plan->data_imag[base + bin] = input_imag[bin];
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
    /* LMUL=m4 leaves eight architectural register groups. Issue independent
     * loads first, then reuse groups after both complex products consume the
     * odd inputs. */
    vfloat32m4_t odd_r = __riscv_vle32_v_f32m4(odd_real, vl);
    vfloat32m4_t odd_i = __riscv_vle32_v_f32m4(odd_imag, vl);

    vfloat32m4_t product_r = __riscv_vfmul_vf_f32m4(odd_r, twiddle_real, vl);
    vfloat32m4_t product_i = __riscv_vfmul_vf_f32m4(odd_i, twiddle_real, vl);
    product_r = __riscv_vfnmsac_vf_f32m4(product_r, twiddle_imag, odd_i, vl);
    product_i = __riscv_vfmacc_vf_f32m4(product_i, twiddle_imag, odd_r, vl);

    vfloat32m4_t even_r = __riscv_vle32_v_f32m4(even_real, vl);
    vfloat32m4_t upper_r = __riscv_vfadd_vv_f32m4(even_r, product_r, vl);
    vfloat32m4_t lower_r = __riscv_vfsub_vv_f32m4(even_r, product_r, vl);
    __riscv_vse32_v_f32m4(even_real, upper_r, vl);
    __riscv_vse32_v_f32m4(odd_real, lower_r, vl);

    vfloat32m4_t even_i = __riscv_vle32_v_f32m4(even_imag, vl);
    vfloat32m4_t upper_i = __riscv_vfadd_vv_f32m4(even_i, product_i, vl);
    vfloat32m4_t lower_i = __riscv_vfsub_vv_f32m4(even_i, product_i, vl);
    __riscv_vse32_v_f32m4(even_imag, upper_i, vl);
    __riscv_vse32_v_f32m4(odd_imag, lower_i, vl);
}

static inline void butterfly_unity_rvv_f32(float *restrict even_real,
                                           float *restrict even_imag,
                                           float *restrict odd_real,
                                           float *restrict odd_imag,
                                           size_t vl)
{
    vfloat32m4_t odd_r = __riscv_vle32_v_f32m4(odd_real, vl);
    vfloat32m4_t even_r = __riscv_vle32_v_f32m4(even_real, vl);
    vfloat32m4_t upper_r = __riscv_vfadd_vv_f32m4(even_r, odd_r, vl);
    vfloat32m4_t lower_r = __riscv_vfsub_vv_f32m4(even_r, odd_r, vl);
    __riscv_vse32_v_f32m4(even_real, upper_r, vl);
    __riscv_vse32_v_f32m4(odd_real, lower_r, vl);

    vfloat32m4_t odd_i = __riscv_vle32_v_f32m4(odd_imag, vl);
    vfloat32m4_t even_i = __riscv_vle32_v_f32m4(even_imag, vl);
    vfloat32m4_t upper_i = __riscv_vfadd_vv_f32m4(even_i, odd_i, vl);
    vfloat32m4_t lower_i = __riscv_vfsub_vv_f32m4(even_i, odd_i, vl);
    __riscv_vse32_v_f32m4(even_imag, upper_i, vl);
    __riscv_vse32_v_f32m4(odd_imag, lower_i, vl);
}

static inline void butterfly_pair_rvv_f32(float *restrict even_real,
                                          float *restrict even_imag,
                                          float *restrict odd_real,
                                          float *restrict odd_imag,
                                          float twiddle_real,
                                          float twiddle_imag,
                                          size_t vl)
{
    float *even_real_1 = even_real + vl;
    float *even_imag_1 = even_imag + vl;
    float *odd_real_1 = odd_real + vl;
    float *odd_imag_1 = odd_imag + vl;

    vfloat32m4_t odd_r0 = __riscv_vle32_v_f32m4(odd_real, vl);
    vfloat32m4_t odd_i0 = __riscv_vle32_v_f32m4(odd_imag, vl);
    vfloat32m4_t odd_r1 = __riscv_vle32_v_f32m4(odd_real_1, vl);
    vfloat32m4_t odd_i1 = __riscv_vle32_v_f32m4(odd_imag_1, vl);

    vfloat32m4_t product_r0 = __riscv_vfmul_vf_f32m4(odd_r0, twiddle_real, vl);
    vfloat32m4_t product_i0 = __riscv_vfmul_vf_f32m4(odd_i0, twiddle_real, vl);
    vfloat32m4_t product_r1 = __riscv_vfmul_vf_f32m4(odd_r1, twiddle_real, vl);
    vfloat32m4_t product_i1 = __riscv_vfmul_vf_f32m4(odd_i1, twiddle_real, vl);
    product_r0 = __riscv_vfnmsac_vf_f32m4(product_r0, twiddle_imag, odd_i0, vl);
    product_i0 = __riscv_vfmacc_vf_f32m4(product_i0, twiddle_imag, odd_r0, vl);
    product_r1 = __riscv_vfnmsac_vf_f32m4(product_r1, twiddle_imag, odd_i1, vl);
    product_i1 = __riscv_vfmacc_vf_f32m4(product_i1, twiddle_imag, odd_r1, vl);

    vfloat32m4_t even_r0 = __riscv_vle32_v_f32m4(even_real, vl);
    vfloat32m4_t even_r1 = __riscv_vle32_v_f32m4(even_real_1, vl);
    vfloat32m4_t upper_r0 = __riscv_vfadd_vv_f32m4(even_r0, product_r0, vl);
    vfloat32m4_t lower_r0 = __riscv_vfsub_vv_f32m4(even_r0, product_r0, vl);
    vfloat32m4_t upper_r1 = __riscv_vfadd_vv_f32m4(even_r1, product_r1, vl);
    vfloat32m4_t lower_r1 = __riscv_vfsub_vv_f32m4(even_r1, product_r1, vl);
    __riscv_vse32_v_f32m4(even_real, upper_r0, vl);
    __riscv_vse32_v_f32m4(odd_real, lower_r0, vl);
    __riscv_vse32_v_f32m4(even_real_1, upper_r1, vl);
    __riscv_vse32_v_f32m4(odd_real_1, lower_r1, vl);

    vfloat32m4_t even_i0 = __riscv_vle32_v_f32m4(even_imag, vl);
    vfloat32m4_t even_i1 = __riscv_vle32_v_f32m4(even_imag_1, vl);
    vfloat32m4_t upper_i0 = __riscv_vfadd_vv_f32m4(even_i0, product_i0, vl);
    vfloat32m4_t lower_i0 = __riscv_vfsub_vv_f32m4(even_i0, product_i0, vl);
    vfloat32m4_t upper_i1 = __riscv_vfadd_vv_f32m4(even_i1, product_i1, vl);
    vfloat32m4_t lower_i1 = __riscv_vfsub_vv_f32m4(even_i1, product_i1, vl);
    __riscv_vse32_v_f32m4(even_imag, upper_i0, vl);
    __riscv_vse32_v_f32m4(odd_imag, lower_i0, vl);
    __riscv_vse32_v_f32m4(even_imag_1, upper_i1, vl);
    __riscv_vse32_v_f32m4(odd_imag_1, lower_i1, vl);
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
    for (; batch_count - batch >= 2 * vlmax; batch += 2 * vlmax) {
        butterfly_pair_rvv_f32(even_real + batch, even_imag + batch,
                               odd_real + batch, odd_imag + batch,
                               twiddle_real, twiddle_imag, vlmax);
    }
    for (; batch_count - batch >= vlmax; batch += vlmax) {
        butterfly_rvv_f32(even_real + batch, even_imag + batch,
                          odd_real + batch, odd_imag + batch,
                          twiddle_real, twiddle_imag, vlmax);
    }
    if (batch < batch_count) {
        size_t remaining = batch_count - batch;
        size_t tail_vl = __riscv_vsetvl_e32m4(remaining);
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
        size_t tail_vl = __riscv_vsetvl_e32m4(remaining);
        if (tail_vl == 0 || tail_vl > remaining) return 0;
        butterfly_unity_rvv_f32(even_real + batch, even_imag + batch,
                                odd_real + batch, odd_imag + batch, tail_vl);
    }
    return 1;
}

static int fft_bit_reverse_rvv_f32(const fft_plan_f32_t *plan)
{
    /* BIN-MAJOR: swap two contiguous rows, each containing all 64 batches. */
    for (size_t index = 0; index < plan->size; ++index) {
        size_t reversed = reverse_bits((unsigned)index, plan->stage_count);
        if (reversed <= index) continue;
        size_t a = data_offset(plan, index, 0);
        size_t b = data_offset(plan, reversed, 0);
        if (!swap_rows_e32m1(&plan->data_real[a], &plan->data_imag[a],
                             &plan->data_real[b], &plan->data_imag[b],
                             plan->batch_count)) return 0;
    }
    return 1;
}

static int fft_bit_reverse_batch_major_f32(const fft_plan_f32_t *plan)
{
    /* BATCH-MAJOR: bit-reverse each batch's contiguous FFT independently. */
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t bin = 0; bin < plan->size; ++bin) {
            size_t reversed = reverse_bits((unsigned)bin, plan->stage_count);
            if (reversed <= bin) continue;
            float tmp = plan->data_real[base + bin];
            plan->data_real[base + bin] = plan->data_real[base + reversed];
            plan->data_real[base + reversed] = tmp;
            tmp = plan->data_imag[base + bin];
            plan->data_imag[base + bin] = plan->data_imag[base + reversed];
            plan->data_imag[base + reversed] = tmp;
        }
    }
    return 1;
}

static int fft_butterflies_rvv_f32(const fft_plan_f32_t *plan)
{
    const size_t vl = __riscv_vsetvl_e32m4(plan->batch_count);
    if (vl == 0) return 0;

    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;

        /* Keep the unity case out of the block/batch hot loops. */
        for (size_t block = 0; block < plan->size; block += block_size) {
            size_t even_base = data_offset(plan, block, 0);
            size_t odd_base = data_offset(plan, block + half, 0);
            if (!butterfly_unity_batches_rvv_f32(
                    &plan->data_real[even_base],
                    &plan->data_imag[even_base],
                    &plan->data_real[odd_base],
                    &plan->data_imag[odd_base],
                    plan->batch_count, vl)) return 0;
        }

        for (size_t offset = 1; offset < half; ++offset) {
            float wr = plan->twiddle_real[stage_base + offset];
            float wi = plan->twiddle_imag[stage_base + offset];

            for (size_t block = 0; block < plan->size;
                 block += block_size) {
                size_t even = block + offset;
                size_t odd = even + half;
                size_t even_base = data_offset(plan, even, 0);
                size_t odd_base = data_offset(plan, odd, 0);
                if (!butterfly_batches_rvv_f32(
                        &plan->data_real[even_base],
                        &plan->data_imag[even_base],
                        &plan->data_real[odd_base],
                        &plan->data_imag[odd_base], wr, wi,
                        plan->batch_count, vl)) return 0;
          }
        }
    }
    return 1;
}

#define DEFINE_BATCH_MAJOR_F32_VARIANT(SFX, VTYPE, VLOAD, VSTORE, VMUL, \
                                        VNMSAC, VMACC, VADD, VSUB, VSETVL) \
static int fft_butterflies_batch_major_##SFX(const fft_plan_f32_t *plan)    \
{                                                                          \
    /* BATCH-MAJOR: process one FFT and vectorize consecutive bin offsets. */ \
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {           \
        size_t base = batch * plan->size;                                   \
        float *real = &plan->data_real[base];                               \
        float *imag = &plan->data_imag[base];                               \
        for (size_t bs = 2; bs <= plan->size; bs <<= 1) {                  \
            size_t half = bs >> 1, stage_base = half - 1;                  \
            for (size_t block = 0; block < plan->size; block += bs) {      \
                float er = real[block], ei = imag[block];                   \
                float or_ = real[block + half], oi = imag[block + half];   \
                real[block] = er + or_; real[block + half] = er - or_;     \
                imag[block] = ei + oi; imag[block + half] = ei - oi;       \
                for (size_t off = 1; off < half;) {                         \
                    size_t vl = VSETVL(half - off);                         \
                    if (vl == 0 || vl > half - off) return 0;               \
                    VTYPE xr = VLOAD(&real[block + half + off], vl);        \
                    VTYPE xi = VLOAD(&imag[block + half + off], vl);        \
                    VTYPE wr = VLOAD(&plan->twiddle_real[stage_base + off], vl); \
                    VTYPE wi = VLOAD(&plan->twiddle_imag[stage_base + off], vl); \
                    VTYPE pr = VMUL(xr, wr, vl);                            \
                    VTYPE pi = VMUL(xi, wr, vl);                            \
                    pr = VNMSAC(pr, wi, xi, vl);                            \
                    pi = VMACC(pi, wi, xr, vl);                             \
                    VTYPE ar = VLOAD(&real[block + off], vl);               \
                    VTYPE ai = VLOAD(&imag[block + off], vl);               \
                    VTYPE ur = VADD(ar, pr, vl), lr = VSUB(ar, pr, vl);     \
                    VTYPE ui = VADD(ai, pi, vl), li = VSUB(ai, pi, vl);     \
                    VSTORE(&real[block + off], ur, vl);                     \
                    VSTORE(&real[block + half + off], lr, vl);              \
                    VSTORE(&imag[block + off], ui, vl);                     \
                    VSTORE(&imag[block + half + off], li, vl);              \
                    off += vl;                                              \
                }                                                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

DEFINE_BATCH_MAJOR_F32_VARIANT(m2, vfloat32m2_t,
    __riscv_vle32_v_f32m2, __riscv_vse32_v_f32m2,
    __riscv_vfmul_vv_f32m2, __riscv_vfnmsac_vv_f32m2,
    __riscv_vfmacc_vv_f32m2, __riscv_vfadd_vv_f32m2,
    __riscv_vfsub_vv_f32m2, __riscv_vsetvl_e32m2)
DEFINE_BATCH_MAJOR_F32_VARIANT(m4, vfloat32m4_t,
    __riscv_vle32_v_f32m4, __riscv_vse32_v_f32m4,
    __riscv_vfmul_vv_f32m4, __riscv_vfnmsac_vv_f32m4,
    __riscv_vfmacc_vv_f32m4, __riscv_vfadd_vv_f32m4,
    __riscv_vfsub_vv_f32m4, __riscv_vsetvl_e32m4)
DEFINE_BATCH_MAJOR_F32_VARIANT(m8, vfloat32m8_t,
    __riscv_vle32_v_f32m8, __riscv_vse32_v_f32m8,
    __riscv_vfmul_vv_f32m8, __riscv_vfnmsac_vv_f32m8,
    __riscv_vfmacc_vv_f32m8, __riscv_vfadd_vv_f32m8,
    __riscv_vfsub_vv_f32m8, __riscv_vsetvl_e32m8)

#define DEFINE_SIMPLE_LMUL_VARIANT(SFX, VTYPE, VLOAD, VSTORE, VMUL, VNMSAC, \
                                   VMACC, VADD, VSUB, VSETVL)              \
static inline void butterfly_##SFX(float *er, float *ei, float *or_,       \
                                    float *oi, float wr, float wi, size_t vl) \
{                                                                          \
    VTYPE xr = VLOAD(or_, vl);                                              \
    VTYPE xi = VLOAD(oi, vl);                                               \
    VTYPE pr = VMUL(xr, wr, vl);                                            \
    VTYPE pi = VMUL(xi, wr, vl);                                            \
    pr = VNMSAC(pr, wi, xi, vl);                                            \
    pi = VMACC(pi, wi, xr, vl);                                             \
    VTYPE ar = VLOAD(er, vl);                                               \
    VTYPE ur = VADD(ar, pr, vl);                                            \
    VTYPE lr = VSUB(ar, pr, vl);                                            \
    VSTORE(er, ur, vl); VSTORE(or_, lr, vl);                                \
    VTYPE ai = VLOAD(ei, vl);                                               \
    VTYPE ui = VADD(ai, pi, vl);                                            \
    VTYPE li = VSUB(ai, pi, vl);                                            \
    VSTORE(ei, ui, vl); VSTORE(oi, li, vl);                                 \
}                                                                          \
static inline void butterfly_unity_##SFX(float *er, float *ei, float *or_, \
                                          float *oi, size_t vl)             \
{                                                                          \
    VTYPE xr = VLOAD(or_, vl); VTYPE ar = VLOAD(er, vl);                    \
    VTYPE ur = VADD(ar, xr, vl); VTYPE lr = VSUB(ar, xr, vl);               \
    VSTORE(er, ur, vl); VSTORE(or_, lr, vl);                                \
    VTYPE xi = VLOAD(oi, vl); VTYPE ai = VLOAD(ei, vl);                     \
    VTYPE ui = VADD(ai, xi, vl); VTYPE li = VSUB(ai, xi, vl);               \
    VSTORE(ei, ui, vl); VSTORE(oi, li, vl);                                 \
}                                                                          \
static int fft_butterflies_##SFX(const fft_plan_f32_t *plan)                \
{                                                                          \
    size_t vl = VSETVL(plan->batch_count);                                  \
    if (vl == 0) return 0;                                                   \
    for (size_t bs = 2; bs <= plan->size; bs <<= 1) {                       \
        size_t half = bs >> 1, stage_base = half - 1;                       \
        for (size_t block = 0; block < plan->size; block += bs) {           \
            size_t eb = data_offset(plan, block, 0);                        \
            size_t ob = data_offset(plan, block + half, 0);                 \
            size_t batch = 0;                                               \
            for (; plan->batch_count - batch >= vl; batch += vl)            \
                butterfly_unity_##SFX(&plan->data_real[eb + batch],         \
                    &plan->data_imag[eb + batch],                           \
                    &plan->data_real[ob + batch],                           \
                    &plan->data_imag[ob + batch], vl);                      \
            if (batch < plan->batch_count) {                                \
                size_t tvl = VSETVL(plan->batch_count - batch);             \
                butterfly_unity_##SFX(&plan->data_real[eb + batch],         \
                    &plan->data_imag[eb + batch],                           \
                    &plan->data_real[ob + batch],                           \
                    &plan->data_imag[ob + batch], tvl);                     \
            }                                                               \
        }                                                                   \
        for (size_t off = 1; off < half; ++off) {                           \
            float wr = plan->twiddle_real[stage_base + off];                \
            float wi = plan->twiddle_imag[stage_base + off];                \
            for (size_t block = 0; block < plan->size; block += bs) {       \
                size_t eb = data_offset(plan, block + off, 0);              \
                size_t ob = data_offset(plan, block + off + half, 0);       \
                size_t batch = 0;                                           \
                for (; plan->batch_count - batch >= vl; batch += vl)        \
                    butterfly_##SFX(&plan->data_real[eb + batch],           \
                        &plan->data_imag[eb + batch],                       \
                        &plan->data_real[ob + batch],                       \
                        &plan->data_imag[ob + batch], wr, wi, vl);          \
                if (batch < plan->batch_count) {                            \
                    size_t tvl = VSETVL(plan->batch_count - batch);         \
                    butterfly_##SFX(&plan->data_real[eb + batch],           \
                        &plan->data_imag[eb + batch],                       \
                        &plan->data_real[ob + batch],                       \
                        &plan->data_imag[ob + batch], wr, wi, tvl);         \
                }                                                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
    return 1;                                                               \
}

DEFINE_SIMPLE_LMUL_VARIANT(m2, vfloat32m2_t, __riscv_vle32_v_f32m2,
    __riscv_vse32_v_f32m2, __riscv_vfmul_vf_f32m2,
    __riscv_vfnmsac_vf_f32m2, __riscv_vfmacc_vf_f32m2,
    __riscv_vfadd_vv_f32m2, __riscv_vfsub_vv_f32m2, __riscv_vsetvl_e32m2)
DEFINE_SIMPLE_LMUL_VARIANT(m8, vfloat32m8_t, __riscv_vle32_v_f32m8,
    __riscv_vse32_v_f32m8, __riscv_vfmul_vf_f32m8,
    __riscv_vfnmsac_vf_f32m8, __riscv_vfmacc_vf_f32m8,
    __riscv_vfadd_vv_f32m8, __riscv_vfsub_vv_f32m8, __riscv_vsetvl_e32m8)

static inline void butterfly_two_stages_m2(
    float *r0p, float *i0p, float *r1p, float *i1p,
    float *r2p, float *i2p, float *r3p, float *i3p,
    float wa_r, float wa_i, float wb0_r, float wb0_i,
    float wb1_r, float wb1_i, size_t vl)
{
    /* First radix-2 stage. Keep all four complex intermediate rows in m2
     * registers; no stage-boundary stores are performed. */
    vfloat32m2_t r0 = __riscv_vle32_v_f32m2(r0p, vl);
    vfloat32m2_t i0 = __riscv_vle32_v_f32m2(i0p, vl);
    vfloat32m2_t r1 = __riscv_vle32_v_f32m2(r1p, vl);
    vfloat32m2_t i1 = __riscv_vle32_v_f32m2(i1p, vl);
    vfloat32m2_t t1r = __riscv_vfmul_vf_f32m2(r1, wa_r, vl);
    vfloat32m2_t t1i = __riscv_vfmul_vf_f32m2(i1, wa_r, vl);
    t1r = __riscv_vfnmsac_vf_f32m2(t1r, wa_i, i1, vl);
    t1i = __riscv_vfmacc_vf_f32m2(t1i, wa_i, r1, vl);
    vfloat32m2_t y0r = __riscv_vfadd_vv_f32m2(r0, t1r, vl);
    vfloat32m2_t y1r = __riscv_vfsub_vv_f32m2(r0, t1r, vl);
    vfloat32m2_t y0i = __riscv_vfadd_vv_f32m2(i0, t1i, vl);
    vfloat32m2_t y1i = __riscv_vfsub_vv_f32m2(i0, t1i, vl);

    vfloat32m2_t r2 = __riscv_vle32_v_f32m2(r2p, vl);
    vfloat32m2_t i2 = __riscv_vle32_v_f32m2(i2p, vl);
    vfloat32m2_t r3 = __riscv_vle32_v_f32m2(r3p, vl);
    vfloat32m2_t i3 = __riscv_vle32_v_f32m2(i3p, vl);
    vfloat32m2_t t3r = __riscv_vfmul_vf_f32m2(r3, wa_r, vl);
    vfloat32m2_t t3i = __riscv_vfmul_vf_f32m2(i3, wa_r, vl);
    t3r = __riscv_vfnmsac_vf_f32m2(t3r, wa_i, i3, vl);
    t3i = __riscv_vfmacc_vf_f32m2(t3i, wa_i, r3, vl);
    vfloat32m2_t y2r = __riscv_vfadd_vv_f32m2(r2, t3r, vl);
    vfloat32m2_t y3r = __riscv_vfsub_vv_f32m2(r2, t3r, vl);
    vfloat32m2_t y2i = __riscv_vfadd_vv_f32m2(i2, t3i, vl);
    vfloat32m2_t y3i = __riscv_vfsub_vv_f32m2(i2, t3i, vl);

    /* Second radix-2 stage. Its two offsets use different twiddles. */
    vfloat32m2_t q0r = __riscv_vfmul_vf_f32m2(y2r, wb0_r, vl);
    vfloat32m2_t q0i = __riscv_vfmul_vf_f32m2(y2i, wb0_r, vl);
    q0r = __riscv_vfnmsac_vf_f32m2(q0r, wb0_i, y2i, vl);
    q0i = __riscv_vfmacc_vf_f32m2(q0i, wb0_i, y2r, vl);
    vfloat32m2_t z0r = __riscv_vfadd_vv_f32m2(y0r, q0r, vl);
    vfloat32m2_t z2r = __riscv_vfsub_vv_f32m2(y0r, q0r, vl);
    vfloat32m2_t z0i = __riscv_vfadd_vv_f32m2(y0i, q0i, vl);
    vfloat32m2_t z2i = __riscv_vfsub_vv_f32m2(y0i, q0i, vl);

    vfloat32m2_t q1r = __riscv_vfmul_vf_f32m2(y3r, wb1_r, vl);
    vfloat32m2_t q1i = __riscv_vfmul_vf_f32m2(y3i, wb1_r, vl);
    q1r = __riscv_vfnmsac_vf_f32m2(q1r, wb1_i, y3i, vl);
    q1i = __riscv_vfmacc_vf_f32m2(q1i, wb1_i, y3r, vl);
    vfloat32m2_t z1r = __riscv_vfadd_vv_f32m2(y1r, q1r, vl);
    vfloat32m2_t z3r = __riscv_vfsub_vv_f32m2(y1r, q1r, vl);
    vfloat32m2_t z1i = __riscv_vfadd_vv_f32m2(y1i, q1i, vl);
    vfloat32m2_t z3i = __riscv_vfsub_vv_f32m2(y1i, q1i, vl);

    __riscv_vse32_v_f32m2(r0p, z0r, vl);
    __riscv_vse32_v_f32m2(i0p, z0i, vl);
    __riscv_vse32_v_f32m2(r1p, z1r, vl);
    __riscv_vse32_v_f32m2(i1p, z1i, vl);
    __riscv_vse32_v_f32m2(r2p, z2r, vl);
    __riscv_vse32_v_f32m2(i2p, z2i, vl);
    __riscv_vse32_v_f32m2(r3p, z3r, vl);
    __riscv_vse32_v_f32m2(i3p, z3i, vl);
}

static int fft_butterflies_m2_stage_fused(const fft_plan_f32_t *plan)
{
    if ((plan->stage_count & 1u) != 0u) return 0;
    for (size_t bs = 2; bs <= plan->size; bs <<= 2) {
        size_t half = bs >> 1;
        size_t next_bs = bs << 1;
        size_t stage_a = half - 1;
        size_t stage_b = bs - 1;
        for (size_t block = 0; block < plan->size; block += next_bs) {
            for (size_t off = 0; off < half; ++off) {
                float wa_r = plan->twiddle_real[stage_a + off];
                float wa_i = plan->twiddle_imag[stage_a + off];
                float wb0_r = plan->twiddle_real[stage_b + off];
                float wb0_i = plan->twiddle_imag[stage_b + off];
                float wb1_r = plan->twiddle_real[stage_b + half + off];
                float wb1_i = plan->twiddle_imag[stage_b + half + off];
                size_t bins[4] = {block + off, block + half + off,
                                  block + bs + off,
                                  block + bs + half + off};
                for (size_t batch = 0; batch < plan->batch_count;) {
                    size_t vl = __riscv_vsetvl_e32m2(
                        plan->batch_count - batch);
                    if (vl == 0) return 0;
                    size_t p0 = data_offset(plan, bins[0], batch);
                    size_t p1 = data_offset(plan, bins[1], batch);
                    size_t p2 = data_offset(plan, bins[2], batch);
                    size_t p3 = data_offset(plan, bins[3], batch);
                    butterfly_two_stages_m2(
                        &plan->data_real[p0], &plan->data_imag[p0],
                        &plan->data_real[p1], &plan->data_imag[p1],
                        &plan->data_real[p2], &plan->data_imag[p2],
                        &plan->data_real[p3], &plan->data_imag[p3],
                        wa_r, wa_i, wb0_r, wb0_i, wb1_r, wb1_i, vl);
                    batch += vl;
                }
            }
        }
    }
    return 1;
}

#define DEFINE_STAGE_FUSED_VARIANT(SFX, VTYPE, VLOAD, VSTORE, VMUL, VNMSAC, \
                                   VMACC, VADD, VSUB, VSETVL)              \
static inline void butterfly_two_stages_##SFX(                             \
    float *r0p, float *i0p, float *r1p, float *i1p,                        \
    float *r2p, float *i2p, float *r3p, float *i3p,                        \
    float ar, float ai, float b0r, float b0i,                              \
    float b1r, float b1i, size_t vl)                                       \
{                                                                          \
    VTYPE r0 = VLOAD(r0p, vl), i0 = VLOAD(i0p, vl);                        \
    VTYPE r1 = VLOAD(r1p, vl), i1 = VLOAD(i1p, vl);                        \
    VTYPE t1r = VMUL(r1, ar, vl), t1i = VMUL(i1, ar, vl);                  \
    t1r = VNMSAC(t1r, ai, i1, vl); t1i = VMACC(t1i, ai, r1, vl);          \
    VTYPE y0r = VADD(r0, t1r, vl), y1r = VSUB(r0, t1r, vl);               \
    VTYPE y0i = VADD(i0, t1i, vl), y1i = VSUB(i0, t1i, vl);               \
    VTYPE r2 = VLOAD(r2p, vl), i2 = VLOAD(i2p, vl);                        \
    VTYPE r3 = VLOAD(r3p, vl), i3 = VLOAD(i3p, vl);                        \
    VTYPE t3r = VMUL(r3, ar, vl), t3i = VMUL(i3, ar, vl);                  \
    t3r = VNMSAC(t3r, ai, i3, vl); t3i = VMACC(t3i, ai, r3, vl);          \
    VTYPE y2r = VADD(r2, t3r, vl), y3r = VSUB(r2, t3r, vl);               \
    VTYPE y2i = VADD(i2, t3i, vl), y3i = VSUB(i2, t3i, vl);               \
    VTYPE q0r = VMUL(y2r, b0r, vl), q0i = VMUL(y2i, b0r, vl);             \
    q0r = VNMSAC(q0r, b0i, y2i, vl); q0i = VMACC(q0i, b0i, y2r, vl);     \
    VTYPE z0r = VADD(y0r, q0r, vl), z2r = VSUB(y0r, q0r, vl);             \
    VTYPE z0i = VADD(y0i, q0i, vl), z2i = VSUB(y0i, q0i, vl);             \
    VTYPE q1r = VMUL(y3r, b1r, vl), q1i = VMUL(y3i, b1r, vl);             \
    q1r = VNMSAC(q1r, b1i, y3i, vl); q1i = VMACC(q1i, b1i, y3r, vl);     \
    VTYPE z1r = VADD(y1r, q1r, vl), z3r = VSUB(y1r, q1r, vl);             \
    VTYPE z1i = VADD(y1i, q1i, vl), z3i = VSUB(y1i, q1i, vl);             \
    VSTORE(r0p, z0r, vl); VSTORE(i0p, z0i, vl);                           \
    VSTORE(r1p, z1r, vl); VSTORE(i1p, z1i, vl);                           \
    VSTORE(r2p, z2r, vl); VSTORE(i2p, z2i, vl);                           \
    VSTORE(r3p, z3r, vl); VSTORE(i3p, z3i, vl);                           \
}                                                                          \
static int fft_butterflies_##SFX##_stage_fused(const fft_plan_f32_t *plan) \
{                                                                          \
    if ((plan->stage_count & 1u) != 0u) return 0;                           \
    for (size_t bs = 2; bs <= plan->size; bs <<= 2) {                      \
        size_t half = bs >> 1, next_bs = bs << 1;                          \
        size_t stage_a = half - 1, stage_b = bs - 1;                       \
        for (size_t block = 0; block < plan->size; block += next_bs)       \
            for (size_t off = 0; off < half; ++off) {                      \
                float ar = plan->twiddle_real[stage_a + off];              \
                float ai = plan->twiddle_imag[stage_a + off];              \
                float b0r = plan->twiddle_real[stage_b + off];             \
                float b0i = plan->twiddle_imag[stage_b + off];             \
                float b1r = plan->twiddle_real[stage_b + half + off];      \
                float b1i = plan->twiddle_imag[stage_b + half + off];      \
                size_t b0 = block + off, b1 = b0 + half;                  \
                size_t b2 = b0 + bs, b3 = b1 + bs;                        \
                for (size_t batch = 0; batch < plan->batch_count;) {       \
                    size_t vl = VSETVL(plan->batch_count - batch);         \
                    if (vl == 0) return 0;                                 \
                    size_t p0 = data_offset(plan, b0, batch);              \
                    size_t p1 = data_offset(plan, b1, batch);              \
                    size_t p2 = data_offset(plan, b2, batch);              \
                    size_t p3 = data_offset(plan, b3, batch);              \
                    butterfly_two_stages_##SFX(                            \
                        &plan->data_real[p0], &plan->data_imag[p0],        \
                        &plan->data_real[p1], &plan->data_imag[p1],        \
                        &plan->data_real[p2], &plan->data_imag[p2],        \
                        &plan->data_real[p3], &plan->data_imag[p3],        \
                        ar, ai, b0r, b0i, b1r, b1i, vl);                   \
                    batch += vl;                                           \
                }                                                          \
            }                                                              \
    }                                                                      \
    return 1;                                                              \
}

DEFINE_STAGE_FUSED_VARIANT(m4, vfloat32m4_t, __riscv_vle32_v_f32m4,
    __riscv_vse32_v_f32m4, __riscv_vfmul_vf_f32m4,
    __riscv_vfnmsac_vf_f32m4, __riscv_vfmacc_vf_f32m4,
    __riscv_vfadd_vv_f32m4, __riscv_vfsub_vv_f32m4,
    __riscv_vsetvl_e32m4)
DEFINE_STAGE_FUSED_VARIANT(m8, vfloat32m8_t, __riscv_vle32_v_f32m8,
    __riscv_vse32_v_f32m8, __riscv_vfmul_vf_f32m8,
    __riscv_vfnmsac_vf_f32m8, __riscv_vfmacc_vf_f32m8,
    __riscv_vfadd_vv_f32m8, __riscv_vfsub_vv_f32m8,
    __riscv_vsetvl_e32m8)

static size_t count_mismatches_f32(const fft_plan_f32_t *plan)
{
    size_t count = 0;
    for (size_t bin = 0; bin < plan->size; ++bin) {
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t pos = data_offset(plan, bin, batch);
            float re = fabsf(plan->data_real[pos] - groundtruth_real[bin]);
            float ie = fabsf(plan->data_imag[pos] - groundtruth_imag[bin]);
            float rr = re / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[bin]));
            float ir = ie / (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[bin]));
            if (!isfinite(plan->data_real[pos]) ||
                !isfinite(plan->data_imag[pos]) || rr > 1.0f || ir > 1.0f)
                ++count;
        }
    }
    return count;
}

static size_t count_mismatches_batch_major_f32(const fft_plan_f32_t *plan)
{
    size_t count = 0;
    for (size_t batch = 0; batch < plan->batch_count; ++batch) {
        size_t base = batch * plan->size;
        for (size_t bin = 0; bin < plan->size; ++bin) {
            size_t pos = base + bin;
            float re = fabsf(plan->data_real[pos] - groundtruth_real[bin]);
            float ie = fabsf(plan->data_imag[pos] - groundtruth_imag[bin]);
            float rr = re /
                (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_real[bin]));
            float ir = ie /
                (FFT_ATOL + FFT_RTOL * fabsf(groundtruth_imag[bin]));
            if (!isfinite(plan->data_real[pos]) ||
                !isfinite(plan->data_imag[pos]) || rr > 1.0f || ir > 1.0f)
                ++count;
        }
    }
    return count;
}

typedef int (*fft_butterfly_fn_f32)(const fft_plan_f32_t *);

static int run_timing_variant_f32(fft_plan_f32_t *plan,
                                  const char *name,
                                  fft_butterfly_fn_f32 function)
{
    uint64_t layout_cycles = 0;
    uint64_t reverse_cycles = 0;
    uint64_t butterfly_cycles = 0;
    int layout_ok = 1, reverse_ok = 1, butterfly_ok = 1;
    for (unsigned run = 0; run < FFT_BENCHMARK_RUNS; ++run) {
        uint64_t layout_start = nn_runtime_read_cycles();
        layout_ok &= fft_plan_load_input_f32(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        reverse_ok &= fft_bit_reverse_rvv_f32(plan);
        uint64_t reverse_end = nn_runtime_read_cycles();
        butterfly_ok &= function(plan);
        uint64_t fft_end = nn_runtime_read_cycles();
        layout_cycles += layout_end - layout_start;
        reverse_cycles += reverse_end - layout_end;
        butterfly_cycles += fft_end - reverse_end;
    }
    layout_cycles /= FFT_BENCHMARK_RUNS;
    reverse_cycles /= FFT_BENCHMARK_RUNS;
    butterfly_cycles /= FFT_BENCHMARK_RUNS;
    size_t mismatches = count_mismatches_f32(plan);
    printf("\nRVV FP32 bin-major variant: %s\n", name);
    printf("benchmark runs: %u\n", FFT_BENCHMARK_RUNS);
    printf("average input layout cycles: "); print_u64_decimal(layout_cycles);
    printf("\naverage bit reversal cycles: "); print_u64_decimal(reverse_cycles);
    printf("\naverage butterfly cycles: "); print_u64_decimal(butterfly_cycles);
    printf("\naverage FFT cycles: ");
    print_u64_decimal(reverse_cycles + butterfly_cycles);
    printf("\naverage butterfly cycles per FFT: ");
    print_u64_decimal(butterfly_cycles / plan->batch_count);
    printf("\naverage FFT cycles per FFT: ");
    print_u64_decimal((reverse_cycles + butterfly_cycles) / plan->batch_count);
    printf("\nmismatched complex points: %u / %u\n", (unsigned)mismatches,
        (unsigned)(plan->size * plan->batch_count));
    printf("FFT FP32 RVV bin-major %s: %s\n", name,
        layout_ok && reverse_ok && butterfly_ok && mismatches == 0 ? "PASS" : "FAIL");
    return layout_ok && reverse_ok && butterfly_ok && mismatches == 0;
}

static int run_batch_major_variant_f32(fft_plan_f32_t *plan,
                                       const char *name,
                                       fft_butterfly_fn_f32 function)
{
    uint64_t layout_cycles = 0;
    uint64_t reverse_cycles = 0;
    uint64_t butterfly_cycles = 0;
    int layout_ok = 1, reverse_ok = 1, butterfly_ok = 1;
    for (unsigned run = 0; run < FFT_BENCHMARK_RUNS; ++run) {
        uint64_t layout_start = nn_runtime_read_cycles();
        layout_ok &= fft_plan_load_input_batch_major_f32(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        reverse_ok &= fft_bit_reverse_batch_major_f32(plan);
        uint64_t reverse_end = nn_runtime_read_cycles();
        butterfly_ok &= function(plan);
        uint64_t fft_end = nn_runtime_read_cycles();
        layout_cycles += layout_end - layout_start;
        reverse_cycles += reverse_end - layout_end;
        butterfly_cycles += fft_end - reverse_end;
    }
    layout_cycles /= FFT_BENCHMARK_RUNS;
    reverse_cycles /= FFT_BENCHMARK_RUNS;
    butterfly_cycles /= FFT_BENCHMARK_RUNS;
    size_t mismatches = count_mismatches_batch_major_f32(plan);
    printf("\nRVV FP32 batch-major variant: %s\n", name);
    printf("benchmark runs: %u\n", FFT_BENCHMARK_RUNS);
    printf("average input layout cycles: "); print_u64_decimal(layout_cycles);
    printf("\naverage bit reversal cycles: "); print_u64_decimal(reverse_cycles);
    printf("\naverage butterfly cycles: "); print_u64_decimal(butterfly_cycles);
    printf("\naverage FFT cycles: ");
    print_u64_decimal(reverse_cycles + butterfly_cycles);
    printf("\naverage butterfly cycles per FFT: ");
    print_u64_decimal(butterfly_cycles / plan->batch_count);
    printf("\naverage FFT cycles per FFT: ");
    print_u64_decimal((reverse_cycles + butterfly_cycles) / plan->batch_count);
    printf("\nmismatched complex points: %u / %u\n", (unsigned)mismatches,
        (unsigned)(plan->size * plan->batch_count));
    printf("FFT FP32 RVV batch-major %s: %s\n", name,
        layout_ok && reverse_ok && butterfly_ok && mismatches == 0 ?
            "PASS" : "FAIL");
    return layout_ok && reverse_ok && butterfly_ok && mismatches == 0;
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
        printf("FFT batched FP32 RVV: FAIL (invalid FFT plan)\n");
        return 1;
    }
    printf("precision: FP32\nFFT size: %u\nstages: %u\nbatches: %u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count);
    printf("twiddle plan cycles: ");
    print_u64_decimal(plan_end - plan_start);
    printf("\n");

    printf("\n===== FP32 BIN-MAJOR [bin][batch] =====\n");
    printf("RVV lanes span batches; twiddle is scalar-broadcast.\n");
    int m2_variant_ok = run_timing_variant_f32(
        &plan, "m2", fft_butterflies_m2);
    int m2_fused_variant_ok = run_timing_variant_f32(
        &plan, "m2-stage-fused", fft_butterflies_m2_stage_fused);
    int m4_fused_variant_ok = run_timing_variant_f32(
        &plan, "m4-stage-fused", fft_butterflies_m4_stage_fused);

    printf("\nRVV FP32 bin-major variant: m4\nbenchmark runs: %u\n",
        FFT_BENCHMARK_RUNS);
    uint64_t layout_cycles = 0, bit_reverse_cycles = 0, butterfly_cycles = 0;
    int layout_vector_length_ok = 1, bit_reverse_ok = 1, vector_length_ok = 1;
    for (unsigned run = 0; run < FFT_BENCHMARK_RUNS; ++run) {
        uint64_t layout_start = nn_runtime_read_cycles();
        layout_vector_length_ok &= fft_plan_load_input_f32(&plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        bit_reverse_ok &= fft_bit_reverse_rvv_f32(&plan);
        uint64_t bit_reverse_end = nn_runtime_read_cycles();
        vector_length_ok &= fft_butterflies_rvv_f32(&plan);
        uint64_t fft_end = nn_runtime_read_cycles();
        layout_cycles += layout_end - layout_start;
        bit_reverse_cycles += bit_reverse_end - layout_end;
        butterfly_cycles += fft_end - bit_reverse_end;
    }
    layout_cycles /= FFT_BENCHMARK_RUNS;
    bit_reverse_cycles /= FFT_BENCHMARK_RUNS;
    butterfly_cycles /= FFT_BENCHMARK_RUNS;
    uint64_t plan_cycles = plan_end - plan_start;
    uint64_t fft_cycles = bit_reverse_cycles + butterfly_cycles;
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
    printf("average input layout cycles: "); print_u64_decimal(layout_cycles);
    printf("\naverage bit reversal cycles: "); print_u64_decimal(bit_reverse_cycles);
    printf("\naverage butterfly cycles: "); print_u64_decimal(butterfly_cycles);
    printf("\naverage FFT cycles: "); print_u64_decimal(fft_cycles);
    printf("\naverage reused-plan total cycles: "); print_u64_decimal(reusable_total_cycles);
    printf("\nfirst-run equivalent cycles: "); print_u64_decimal(first_run_total_cycles);
    printf("\naverage butterfly cycles per FFT: ");
    print_u64_decimal(butterfly_cycles / plan.batch_count);
    printf("\naverage FFT cycles per FFT: "); print_u64_decimal(fft_cycles / plan.batch_count);
    printf("\naverage reused-plan cycles per FFT: ");
    print_u64_decimal(reusable_total_cycles / plan.batch_count);
    printf("\nfirst-run equivalent cycles per FFT: ");
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
    int m8_variant_ok = run_timing_variant_f32(
        &plan, "m8", fft_butterflies_m8);
    int m8_fused_variant_ok = run_timing_variant_f32(
        &plan, "m8-stage-fused", fft_butterflies_m8_stage_fused);

    printf("\n===== FP32 BATCH-MAJOR [batch][bin] =====\n");
    printf("RVV lanes span bins within one FFT; twiddles are vectors.\n");
    int batch_major_m2_ok = run_batch_major_variant_f32(
        &plan, "m2", fft_butterflies_batch_major_m2);
    int batch_major_m4_ok = run_batch_major_variant_f32(
        &plan, "m4", fft_butterflies_batch_major_m4);
    int batch_major_m8_ok = run_batch_major_variant_f32(
        &plan, "m8", fft_butterflies_batch_major_m8);

    if (!layout_vector_length_ok || !bit_reverse_ok || !vector_length_ok) {
        printf("FFT batched FP32 RVV: FAIL (invalid VL returned by vsetvl)\n");
        fft_plan_destroy_f32(&plan);
        return 1;
    }
    if (!m2_variant_ok || !m2_fused_variant_ok || !m4_fused_variant_ok ||
        !m8_variant_ok || !m8_fused_variant_ok || !batch_major_m2_ok ||
        !batch_major_m4_ok || !batch_major_m8_ok ||
        non_finite_count > 0 || max_ratio > 1.0f) {
        printf("FFT batched FP32 RVV: FAIL (atol %.6f, rtol %.6f)\n",
            FFT_ATOL, FFT_RTOL);
        fft_plan_destroy_f32(&plan);
        return 1;
    }
    printf("FFT batched FP32 RVV: PASS (atol %.6f, rtol %.6f)\n",
        FFT_ATOL, FFT_RTOL);
    fft_plan_destroy_f32(&plan);
    return 0;
}

int main(void)
{
    return fft_batched_testbench_run();
}
