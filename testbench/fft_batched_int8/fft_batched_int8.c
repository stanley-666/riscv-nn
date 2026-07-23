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

#define FFT_INT8_BENCHMARK_RUNS 10u

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

/* The variant name is the LMUL of the i32 accumulation result.  Widening
 * keeps the same number of lanes: e8,mf2 -> e16,m1 -> e32,m2 and
 * e8,m1 -> e16,m2 -> e32,m4.  The existing e8,m2 path produces e32,m8. */
#define DEFINE_Q7_BUTTERFLY(NAME, T8, T16, T32, LOAD8, WMUL16, SEXT32, \
                            SUB32, ADD32, NCLIP16, NCLIP8, VAADD8, VASUB8, \
                            STORE8) \
static inline void butterfly_q7_##NAME(int8_t *restrict erp, \
    int8_t *restrict eip, int8_t *restrict orp, int8_t *restrict oip, \
    int8_t wr, int8_t wi, size_t vl) \
{ \
    T8 or_ = LOAD8(orp, vl); \
    T8 oi = LOAD8(oip, vl); \
    T16 p16 = WMUL16(or_, wr, vl); \
    T16 q16 = WMUL16(oi, wi, vl); \
    T32 p32 = SEXT32(p16, vl); \
    T32 q32 = SEXT32(q16, vl); \
    p32 = SUB32(p32, q32, vl); \
    p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
    T8 tr = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
    p16 = WMUL16(oi, wr, vl); \
    q16 = WMUL16(or_, wi, vl); \
    p32 = SEXT32(p16, vl); \
    q32 = SEXT32(q16, vl); \
    p32 = ADD32(p32, q32, vl); \
    p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
    T8 ti = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
    T8 er = LOAD8(erp, vl); \
    T8 ur = VAADD8(er, tr, __RISCV_VXRM_RNU, vl); \
    T8 lr = VASUB8(er, tr, __RISCV_VXRM_RNU, vl); \
    STORE8(erp, ur, vl); STORE8(orp, lr, vl); \
    T8 ei = LOAD8(eip, vl); \
    T8 ui = VAADD8(ei, ti, __RISCV_VXRM_RNU, vl); \
    T8 li = VASUB8(ei, ti, __RISCV_VXRM_RNU, vl); \
    STORE8(eip, ui, vl); STORE8(oip, li, vl); \
} \
static inline void butterfly_unity_q7_##NAME(int8_t *restrict erp, \
    int8_t *restrict eip, int8_t *restrict orp, int8_t *restrict oip, \
    size_t vl) \
{ \
    T8 or_ = LOAD8(orp, vl); T8 er = LOAD8(erp, vl); \
    T8 ur = VAADD8(er, or_, __RISCV_VXRM_RNU, vl); \
    T8 lr = VASUB8(er, or_, __RISCV_VXRM_RNU, vl); \
    STORE8(erp, ur, vl); STORE8(orp, lr, vl); \
    T8 oi = LOAD8(oip, vl); T8 ei = LOAD8(eip, vl); \
    T8 ui = VAADD8(ei, oi, __RISCV_VXRM_RNU, vl); \
    T8 li = VASUB8(ei, oi, __RISCV_VXRM_RNU, vl); \
    STORE8(eip, ui, vl); STORE8(oip, li, vl); \
}

DEFINE_Q7_BUTTERFLY(m2, vint8mf2_t, vint16m1_t, vint32m2_t,
    __riscv_vle8_v_i8mf2, __riscv_vwmul_vx_i16m1, __riscv_vsext_vf2_i32m2,
    __riscv_vsub_vv_i32m2, __riscv_vadd_vv_i32m2,
    __riscv_vnclip_wx_i16m1, __riscv_vnclip_wx_i8mf2,
    __riscv_vaadd_vv_i8mf2, __riscv_vasub_vv_i8mf2, __riscv_vse8_v_i8mf2)
DEFINE_Q7_BUTTERFLY(m4, vint8m1_t, vint16m2_t, vint32m4_t,
    __riscv_vle8_v_i8m1, __riscv_vwmul_vx_i16m2, __riscv_vsext_vf2_i32m4,
    __riscv_vsub_vv_i32m4, __riscv_vadd_vv_i32m4,
    __riscv_vnclip_wx_i16m2, __riscv_vnclip_wx_i8m1,
    __riscv_vaadd_vv_i8m1, __riscv_vasub_vv_i8m1, __riscv_vse8_v_i8m1)

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

static void fft_bit_reverse_rvv_q7(const fft_plan_q7_t *plan)
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
}

static inline void butterfly_batches_q7(int8_t *even_real,
                                         int8_t *even_imag,
                                         int8_t *odd_real,
                                         int8_t *odd_imag,
                                         int8_t wr,
                                         int8_t wi,
                                         size_t batch_count,
                                         size_t vl)
{
    size_t batch = 0;
    for (; batch_count - batch >= vl; batch += vl) {
        butterfly_q7(even_real + batch, even_imag + batch,
                     odd_real + batch, odd_imag + batch, wr, wi, vl);
    }
    if (batch < batch_count) {
        size_t tail_vl = __riscv_vsetvl_e8m2(batch_count - batch);
        butterfly_q7(even_real + batch, even_imag + batch,
                     odd_real + batch, odd_imag + batch, wr, wi, tail_vl);
    }
}

static inline void butterfly_unity_batches_q7(int8_t *even_real,
                                               int8_t *even_imag,
                                               int8_t *odd_real,
                                               int8_t *odd_imag,
                                               size_t batch_count,
                                               size_t vl)
{
    size_t batch = 0;
    for (; batch_count - batch >= vl; batch += vl) {
        butterfly_unity_q7(even_real + batch, even_imag + batch,
                           odd_real + batch, odd_imag + batch, vl);
    }
    if (batch < batch_count) {
        size_t tail_vl = __riscv_vsetvl_e8m2(batch_count - batch);
        butterfly_unity_q7(even_real + batch, even_imag + batch,
                           odd_real + batch, odd_imag + batch, tail_vl);
    }
}

static int fft_butterflies_rvv_q7(const fft_plan_q7_t *plan)
{
    size_t vl = __riscv_vsetvl_e8m2(plan->batch_count);
    if (vl == 0) return 0;

    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) {
        size_t half = block_size >> 1;
        size_t stage_base = half - 1;

        /* The unity twiddle is a separate loop so the block/batch hot loops
         * contain no offset==0 branch. */
        for (size_t block = 0; block < plan->size; block += block_size) {
            size_t even_base = data_offset(plan, block, 0);
            size_t odd_base = data_offset(plan, block + half, 0);
            butterfly_unity_batches_q7(
                &plan->data_real[even_base], &plan->data_imag[even_base],
                &plan->data_real[odd_base], &plan->data_imag[odd_base],
                plan->batch_count, vl);
        }

        for (size_t offset = 1; offset < half; ++offset) {
            int8_t wr = plan->twiddle_real[stage_base + offset];
            int8_t wi = plan->twiddle_imag[stage_base + offset];
            for (size_t block = 0; block < plan->size; block += block_size) {
                size_t even_base = data_offset(plan, block + offset, 0);
                size_t odd_base = data_offset(plan, block + offset + half, 0);
                butterfly_batches_q7(
                    &plan->data_real[even_base], &plan->data_imag[even_base],
                    &plan->data_real[odd_base], &plan->data_imag[odd_base],
                    wr, wi, plan->batch_count, vl);
            }
        }
    }
    return 1;
}

#define DEFINE_Q7_FFT_VARIANT(NAME, SETVL) \
static int fft_butterflies_q7_##NAME(const fft_plan_q7_t *plan) \
{ \
    size_t full_vl = SETVL(plan->batch_count); \
    if (full_vl == 0) return 0; \
    for (size_t block_size = 2; block_size <= plan->size; block_size <<= 1) { \
        size_t half = block_size >> 1; \
        size_t stage_base = half - 1; \
        for (size_t block = 0; block < plan->size; block += block_size) { \
            size_t e = data_offset(plan, block, 0); \
            size_t o = data_offset(plan, block + half, 0); \
            for (size_t batch = 0; batch < plan->batch_count;) { \
                size_t vl = SETVL(plan->batch_count - batch); \
                butterfly_unity_q7_##NAME(&plan->data_real[e + batch], \
                    &plan->data_imag[e + batch], &plan->data_real[o + batch], \
                    &plan->data_imag[o + batch], vl); \
                batch += vl; \
            } \
        } \
        for (size_t offset = 1; offset < half; ++offset) { \
            int8_t wr = plan->twiddle_real[stage_base + offset]; \
            int8_t wi = plan->twiddle_imag[stage_base + offset]; \
            for (size_t block = 0; block < plan->size; block += block_size) { \
                size_t e = data_offset(plan, block + offset, 0); \
                size_t o = data_offset(plan, block + offset + half, 0); \
                for (size_t batch = 0; batch < plan->batch_count;) { \
                    size_t vl = SETVL(plan->batch_count - batch); \
                    butterfly_q7_##NAME(&plan->data_real[e + batch], \
                        &plan->data_imag[e + batch], &plan->data_real[o + batch], \
                        &plan->data_imag[o + batch], wr, wi, vl); \
                    batch += vl; \
                } \
            } \
        } \
    } \
    return 1; \
}

DEFINE_Q7_FFT_VARIANT(m2, __riscv_vsetvl_e8mf2)
DEFINE_Q7_FFT_VARIANT(m4, __riscv_vsetvl_e8m1)

#define DEFINE_Q7_STAGE_FUSED_VARIANT(NAME, T8, T16, T32, LOAD8, WMUL16, \
    SEXT32, SUB32, ADD32, NCLIP16, NCLIP8, VAADD8, VASUB8, STORE8, SETVL) \
static inline void butterfly_two_stages_q7_##NAME( \
    int8_t *r0p, int8_t *i0p, int8_t *r1p, int8_t *i1p, \
    int8_t *r2p, int8_t *i2p, int8_t *r3p, int8_t *i3p, \
    int8_t ar, int8_t ai, int8_t b0r, int8_t b0i, \
    int8_t b1r, int8_t b1i, int unity, size_t vl) \
{ \
    T8 r0 = LOAD8(r0p, vl), i0 = LOAD8(i0p, vl); \
    T8 r1 = LOAD8(r1p, vl), i1 = LOAD8(i1p, vl); \
    T8 r2 = LOAD8(r2p, vl), i2 = LOAD8(i2p, vl); \
    T8 r3 = LOAD8(r3p, vl), i3 = LOAD8(i3p, vl); \
    T8 y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i; \
    if (unity) { \
        y0r = VAADD8(r0, r1, __RISCV_VXRM_RNU, vl); \
        y1r = VASUB8(r0, r1, __RISCV_VXRM_RNU, vl); \
        y0i = VAADD8(i0, i1, __RISCV_VXRM_RNU, vl); \
        y1i = VASUB8(i0, i1, __RISCV_VXRM_RNU, vl); \
        y2r = VAADD8(r2, r3, __RISCV_VXRM_RNU, vl); \
        y3r = VASUB8(r2, r3, __RISCV_VXRM_RNU, vl); \
        y2i = VAADD8(i2, i3, __RISCV_VXRM_RNU, vl); \
        y3i = VASUB8(i2, i3, __RISCV_VXRM_RNU, vl); \
    } else { \
        T16 p16 = WMUL16(r1, ar, vl), q16 = WMUL16(i1, ai, vl); \
        T32 p32 = SEXT32(p16, vl), q32 = SEXT32(q16, vl); \
        p32 = SUB32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        T8 t1r = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
        p16 = WMUL16(i1, ar, vl); q16 = WMUL16(r1, ai, vl); \
        p32 = SEXT32(p16, vl); q32 = SEXT32(q16, vl); \
        p32 = ADD32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        T8 t1i = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
        p16 = WMUL16(r3, ar, vl); q16 = WMUL16(i3, ai, vl); \
        p32 = SEXT32(p16, vl); q32 = SEXT32(q16, vl); \
        p32 = SUB32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        T8 t3r = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
        p16 = WMUL16(i3, ar, vl); q16 = WMUL16(r3, ai, vl); \
        p32 = SEXT32(p16, vl); q32 = SEXT32(q16, vl); \
        p32 = ADD32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        T8 t3i = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
        y0r = VAADD8(r0, t1r, __RISCV_VXRM_RNU, vl); \
        y1r = VASUB8(r0, t1r, __RISCV_VXRM_RNU, vl); \
        y0i = VAADD8(i0, t1i, __RISCV_VXRM_RNU, vl); \
        y1i = VASUB8(i0, t1i, __RISCV_VXRM_RNU, vl); \
        y2r = VAADD8(r2, t3r, __RISCV_VXRM_RNU, vl); \
        y3r = VASUB8(r2, t3r, __RISCV_VXRM_RNU, vl); \
        y2i = VAADD8(i2, t3i, __RISCV_VXRM_RNU, vl); \
        y3i = VASUB8(i2, t3i, __RISCV_VXRM_RNU, vl); \
    } \
    T8 q0r, q0i; \
    if (unity) { \
        q0r = y2r; q0i = y2i; \
    } else { \
        T16 p16 = WMUL16(y2r, b0r, vl), q16 = WMUL16(y2i, b0i, vl); \
        T32 p32 = SEXT32(p16, vl), q32 = SEXT32(q16, vl); \
        p32 = SUB32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        q0r = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
        p16 = WMUL16(y2i, b0r, vl); q16 = WMUL16(y2r, b0i, vl); \
        p32 = SEXT32(p16, vl); q32 = SEXT32(q16, vl); \
        p32 = ADD32(p32, q32, vl); \
        p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
        q0i = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
    } \
    T16 p16 = WMUL16(y3r, b1r, vl), q16 = WMUL16(y3i, b1i, vl); \
    T32 p32 = SEXT32(p16, vl), q32 = SEXT32(q16, vl); \
    p32 = SUB32(p32, q32, vl); \
    p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
    T8 q1r = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
    p16 = WMUL16(y3i, b1r, vl); q16 = WMUL16(y3r, b1i, vl); \
    p32 = SEXT32(p16, vl); q32 = SEXT32(q16, vl); \
    p32 = ADD32(p32, q32, vl); \
    p16 = NCLIP16(p32, 7, __RISCV_VXRM_RNU, vl); \
    T8 q1i = NCLIP8(p16, 0, __RISCV_VXRM_RNU, vl); \
    T8 z0r = VAADD8(y0r, q0r, __RISCV_VXRM_RNU, vl); \
    T8 z2r = VASUB8(y0r, q0r, __RISCV_VXRM_RNU, vl); \
    T8 z0i = VAADD8(y0i, q0i, __RISCV_VXRM_RNU, vl); \
    T8 z2i = VASUB8(y0i, q0i, __RISCV_VXRM_RNU, vl); \
    T8 z1r = VAADD8(y1r, q1r, __RISCV_VXRM_RNU, vl); \
    T8 z3r = VASUB8(y1r, q1r, __RISCV_VXRM_RNU, vl); \
    T8 z1i = VAADD8(y1i, q1i, __RISCV_VXRM_RNU, vl); \
    T8 z3i = VASUB8(y1i, q1i, __RISCV_VXRM_RNU, vl); \
    STORE8(r0p, z0r, vl); STORE8(i0p, z0i, vl); \
    STORE8(r1p, z1r, vl); STORE8(i1p, z1i, vl); \
    STORE8(r2p, z2r, vl); STORE8(i2p, z2i, vl); \
    STORE8(r3p, z3r, vl); STORE8(i3p, z3i, vl); \
} \
static int fft_butterflies_q7_##NAME##_stage_fused( \
    const fft_plan_q7_t *plan) \
{ \
    if ((plan->stage_count & 1u) != 0u) return 0; \
    for (size_t bs = 2; bs <= plan->size; bs <<= 2) { \
        size_t half = bs >> 1, next_bs = bs << 1; \
        size_t stage_a = half - 1, stage_b = bs - 1; \
        for (size_t block = 0; block < plan->size; block += next_bs) \
            for (size_t off = 0; off < half; ++off) { \
                int8_t ar = plan->twiddle_real[stage_a + off]; \
                int8_t ai = plan->twiddle_imag[stage_a + off]; \
                int8_t b0r = plan->twiddle_real[stage_b + off]; \
                int8_t b0i = plan->twiddle_imag[stage_b + off]; \
                int8_t b1r = plan->twiddle_real[stage_b + half + off]; \
                int8_t b1i = plan->twiddle_imag[stage_b + half + off]; \
                size_t b0 = block + off, b1 = b0 + half; \
                size_t b2 = b0 + bs, b3 = b1 + bs; \
                for (size_t batch = 0; batch < plan->batch_count;) { \
                    size_t vl = SETVL(plan->batch_count - batch); \
                    if (vl == 0) return 0; \
                    size_t p0 = data_offset(plan, b0, batch); \
                    size_t p1 = data_offset(plan, b1, batch); \
                    size_t p2 = data_offset(plan, b2, batch); \
                    size_t p3 = data_offset(plan, b3, batch); \
                    butterfly_two_stages_q7_##NAME( \
                        &plan->data_real[p0], &plan->data_imag[p0], \
                        &plan->data_real[p1], &plan->data_imag[p1], \
                        &plan->data_real[p2], &plan->data_imag[p2], \
                        &plan->data_real[p3], &plan->data_imag[p3], \
                        ar, ai, b0r, b0i, b1r, b1i, off == 0, vl); \
                    batch += vl; \
                } \
            } \
    } \
    return 1; \
}

DEFINE_Q7_STAGE_FUSED_VARIANT(m2, vint8mf2_t, vint16m1_t, vint32m2_t,
    __riscv_vle8_v_i8mf2, __riscv_vwmul_vx_i16m1,
    __riscv_vsext_vf2_i32m2, __riscv_vsub_vv_i32m2,
    __riscv_vadd_vv_i32m2, __riscv_vnclip_wx_i16m1,
    __riscv_vnclip_wx_i8mf2, __riscv_vaadd_vv_i8mf2,
    __riscv_vasub_vv_i8mf2, __riscv_vse8_v_i8mf2,
    __riscv_vsetvl_e8mf2)
DEFINE_Q7_STAGE_FUSED_VARIANT(m4, vint8m1_t, vint16m2_t, vint32m4_t,
    __riscv_vle8_v_i8m1, __riscv_vwmul_vx_i16m2,
    __riscv_vsext_vf2_i32m4, __riscv_vsub_vv_i32m4,
    __riscv_vadd_vv_i32m4, __riscv_vnclip_wx_i16m2,
    __riscv_vnclip_wx_i8m1, __riscv_vaadd_vv_i8m1,
    __riscv_vasub_vv_i8m1, __riscv_vse8_v_i8m1,
    __riscv_vsetvl_e8m1)
DEFINE_Q7_STAGE_FUSED_VARIANT(m8, vint8m2_t, vint16m4_t, vint32m8_t,
    __riscv_vle8_v_i8m2, __riscv_vwmul_vx_i16m4,
    __riscv_vsext_vf2_i32m8, __riscv_vsub_vv_i32m8,
    __riscv_vadd_vv_i32m8, __riscv_vnclip_wx_i16m4,
    __riscv_vnclip_wx_i8m2, __riscv_vaadd_vv_i8m2,
    __riscv_vasub_vv_i8m2, __riscv_vse8_v_i8m2,
    __riscv_vsetvl_e8m2)

typedef int (*fft_q7_function_t)(const fft_plan_q7_t *);

static size_t count_mismatches_q7(const fft_plan_q7_t *plan)
{
    size_t mismatches = 0;
    for (size_t bin = 0; bin < plan->size; ++bin)
        for (size_t batch = 0; batch < plan->batch_count; ++batch) {
            size_t pos = data_offset(plan, bin, batch);
            if (plan->data_real[pos] != fft_int8_groundtruth_real[bin] ||
                plan->data_imag[pos] != fft_int8_groundtruth_imag[bin])
                ++mismatches;
        }
    return mismatches;
}

static int run_q7_variant(fft_plan_q7_t *plan, const char *name,
                          fft_q7_function_t function)
{
    uint64_t layout = 0, reverse = 0, butterfly = 0;
    int vl_ok = 1;
    for (unsigned run = 0; run < FFT_INT8_BENCHMARK_RUNS; ++run) {
        uint64_t begin = nn_runtime_read_cycles();
        fft_plan_load_input_q7(plan);
        uint64_t layout_end = nn_runtime_read_cycles();
        fft_bit_reverse_rvv_q7(plan);
        uint64_t reverse_end = nn_runtime_read_cycles();
        vl_ok &= function(plan);
        uint64_t end = nn_runtime_read_cycles();
        layout += layout_end - begin;
        reverse += reverse_end - layout_end;
        butterfly += end - reverse_end;
    }
    layout /= FFT_INT8_BENCHMARK_RUNS;
    reverse /= FFT_INT8_BENCHMARK_RUNS;
    butterfly /= FFT_INT8_BENCHMARK_RUNS;
    size_t mismatches = count_mismatches_q7(plan);
    printf("\nRVV int8 variant: %s\nbenchmark runs: %u\n", name,
        FFT_INT8_BENCHMARK_RUNS);
    printf("average input layout cycles: "); print_u64_decimal(layout);
    printf("\naverage bit reversal cycles: "); print_u64_decimal(reverse);
    printf("\naverage butterfly cycles: "); print_u64_decimal(butterfly);
    printf("\naverage FFT cycles: "); print_u64_decimal(reverse + butterfly);
    printf("\nmismatched complex points: %u / %u\n", (unsigned)mismatches,
        (unsigned)(plan->size * plan->batch_count));
    printf("FFT int8 RVV %s: %s\n", name,
        vl_ok && mismatches == 0 ? "PASS" : "FAIL");
    return vl_ok && mismatches == 0;
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
    uint64_t plan_cycles = plan_end - plan_start;
    printf("FFT size: %u\nstages: %u\nbatches: %u\nscale: 1/%u\n",
        (unsigned)plan.size, plan.stage_count, (unsigned)plan.batch_count,
        (unsigned)plan.size);
    printf("twiddle plan cycles: "); print_u64_decimal(plan_cycles);
    printf("\n");
    int m2_ok = run_q7_variant(&plan, "m2 accumulator",
        fft_butterflies_q7_m2);
    int m2_fused_ok = run_q7_variant(&plan, "m2 accumulator stage-fused",
        fft_butterflies_q7_m2_stage_fused);
    int m4_ok = run_q7_variant(&plan, "m4 accumulator",
        fft_butterflies_q7_m4);
    int m4_fused_ok = run_q7_variant(&plan, "m4 accumulator stage-fused",
        fft_butterflies_q7_m4_stage_fused);
    int m8_ok = run_q7_variant(&plan, "m8 accumulator",
        fft_butterflies_rvv_q7);
    int m8_fused_ok = run_q7_variant(&plan, "m8 accumulator stage-fused",
        fft_butterflies_q7_m8_stage_fused);
    int ok = m2_ok && m2_fused_ok && m4_ok && m4_fused_ok &&
        m8_ok && m8_fused_ok;
    printf("FFT batched int8 RVV: %s (bit-exact Q7 validation)\n",
        ok ? "PASS" : "FAIL");
    fft_plan_destroy_q7(&plan);
    return ok ? 0 : 1;
}

#ifndef BAREMETAL
int main(void)
{
    return fft_batched_int8_testbench_run();
}
#endif
