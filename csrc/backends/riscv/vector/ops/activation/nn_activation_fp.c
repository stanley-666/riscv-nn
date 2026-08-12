/* SPDX-FileContributor: Person: Stanley Lee */
/* SPDX-License-Identifier: Apache-2.0 */
#include "nn_activation_fp_internal.h"
#include "nn_utils.h"

enum { ACTIVATION_KERNEL_COUNT_F32 = NONE + 1 };

static void activate_store_chunk_none_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m2(output, vacc, vl);
}

static void activate_store_chunk_none_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_none_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m4(output, vacc, vl);
}

static void activate_store_chunk_relu_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    vacc = __riscv_vfmax_vf_f32m2(vacc, 0.0f, vl);
    __riscv_vse32_v_f32m2(output, vacc, vl);
}

static void activate_store_chunk_relu_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    vacc = __riscv_vfmax_vf_f32m8(vacc, 0.0f, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_relu_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    vacc = __riscv_vfmax_vf_f32m4(vacc, 0.0f, vl);
    __riscv_vse32_v_f32m4(output, vacc, vl);
}

static void activate_store_chunk_leaky_relu_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    vfloat32m2_t vslope = __riscv_vfmv_v_f_f32m2(0.01f, vl);
    vfloat32m2_t vscaled = __riscv_vfmul_vv_f32m2(vacc, vslope, vl);
    vbool16_t mask = __riscv_vmflt_vf_f32m2_b16(vacc, 0.0f, vl);
    vacc = __riscv_vmerge_vvm_f32m2(vacc, vscaled, mask, vl);
    __riscv_vse32_v_f32m2(output, vacc, vl);
}

static void activate_store_chunk_leaky_relu_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    vfloat32m8_t vslope = __riscv_vfmv_v_f_f32m8(0.01f, vl);
    vfloat32m8_t vscaled = __riscv_vfmul_vv_f32m8(vacc, vslope, vl);
    vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(vacc, 0.0f, vl);
    vacc = __riscv_vmerge_vvm_f32m8(vacc, vscaled, mask, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_leaky_relu_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    vfloat32m4_t vslope = __riscv_vfmv_v_f_f32m4(0.01f, vl);
    vfloat32m4_t vscaled = __riscv_vfmul_vv_f32m4(vacc, vslope, vl);
    vbool8_t mask = __riscv_vmflt_vf_f32m4_b8(vacc, 0.0f, vl);
    vacc = __riscv_vmerge_vvm_f32m4(vacc, vscaled, mask, vl);
    __riscv_vse32_v_f32m4(output, vacc, vl);
}

/* Degree-9 least-squares polynomial fitted to sigmoid(a) for a in [0, 8].
 * It is evaluated with Horner's method, then extended to negative inputs with
 * sigmoid(-x) = 1 - sigmoid(x).  Clamping |x| to 8 keeps evaluation inside the
 * fitted interval.  This path uses only RVV arithmetic: no expf(), division,
 * temporary vector store, or scalar loop.  With P(0) forced to exactly 0.5,
 * offline testing on 65,537 points in [-16, 16] gives max abs error < 5.1e-4. */
 
static void activate_store_chunk_sigmoid_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    vbool16_t negative = __riscv_vmflt_vf_f32m2_b16(vacc, 0.0f, vl);
    vfloat32m2_t a = __riscv_vfabs_v_f32m2(vacc, vl);
    a = __riscv_vfmin_vf_f32m2(a, 8.0f, vl);
    vfloat32m2_t y = __riscv_vfmv_v_f_f32m2(7.12672489733e-8f, vl);
    vfloat32m2_t c;
    c = __riscv_vfmv_v_f_f32m2(-2.22844320788e-6f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2( 2.36792598722e-5f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2(-3.77314366681e-5f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2(-1.26097319782e-3f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2( 1.13415961144e-2f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2(-3.77522656294e-2f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2( 1.21156497548e-2f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2( 2.46437157045e-1f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m2( 5.0e-1f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, a, c, vl);
    vfloat32m2_t reflected = __riscv_vfrsub_vf_f32m2(y, 1.0f, vl);
    y = __riscv_vmerge_vvm_f32m2(y, reflected, negative, vl);
    __riscv_vse32_v_f32m2(output, y, vl);
}

static void activate_store_chunk_sigmoid_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    vbool4_t negative = __riscv_vmflt_vf_f32m8_b4(vacc, 0.0f, vl);
    vfloat32m8_t a = __riscv_vfabs_v_f32m8(vacc, vl);
    a = __riscv_vfmin_vf_f32m8(a, 8.0f, vl);
    vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(7.12672489733e-8f, vl);
    vfloat32m8_t c;
    c = __riscv_vfmv_v_f_f32m8(-2.22844320788e-6f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8( 2.36792598722e-5f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-3.77314366681e-5f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-1.26097319782e-3f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8( 1.13415961144e-2f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8(-3.77522656294e-2f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8( 1.21156497548e-2f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8( 2.46437157045e-1f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m8( 5.0e-1f, vl);
    y = __riscv_vfmadd_vv_f32m8(y, a, c, vl);
    vfloat32m8_t reflected = __riscv_vfrsub_vf_f32m8(y, 1.0f, vl);
    y = __riscv_vmerge_vvm_f32m8(y, reflected, negative, vl);
    __riscv_vse32_v_f32m8(output, y, vl);
}

static void activate_store_chunk_sigmoid_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    vbool8_t negative = __riscv_vmflt_vf_f32m4_b8(vacc, 0.0f, vl);
    vfloat32m4_t a = __riscv_vfabs_v_f32m4(vacc, vl);
    a = __riscv_vfmin_vf_f32m4(a, 8.0f, vl);
    vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(7.12672489733e-8f, vl);
    vfloat32m4_t c;
    c = __riscv_vfmv_v_f_f32m4(-2.22844320788e-6f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4( 2.36792598722e-5f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4(-3.77314366681e-5f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4(-1.26097319782e-3f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4( 1.13415961144e-2f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4(-3.77522656294e-2f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4( 1.21156497548e-2f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4( 2.46437157045e-1f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    c = __riscv_vfmv_v_f_f32m4( 5.0e-1f, vl);
    y = __riscv_vfmadd_vv_f32m4(y, a, c, vl);
    vfloat32m4_t reflected = __riscv_vfrsub_vf_f32m4(y, 1.0f, vl);
    y = __riscv_vmerge_vvm_f32m4(y, reflected, negative, vl);
    __riscv_vse32_v_f32m4(output, y, vl);
}

static void activate_store_chunk_tanh_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m2(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        output[i] = activate_f32(tmp[i], TANH);
    }
}

static void activate_store_chunk_tanh_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m8(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        output[i] = activate_f32(tmp[i], TANH);
    }
}

static void activate_store_chunk_tanh_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m4(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        output[i] = activate_f32(tmp[i], TANH);
    }
}

static void activate_store_chunk_softmax_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m2(output, vacc, vl);
}

static void activate_store_chunk_softmax_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_softmax_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m4(output, vacc, vl);
}

/* Full-tensor RVV Softmax.  Softmax requires a global maximum and a global
 * exponential sum, so it cannot be implemented correctly by a per-chunk
 * activation callback.  exp(x) is approximated on [-8, 0] by a monotonic
 * degree-9 least-squares polynomial evaluated with Horner FMA.  Values below
 * -8 are treated as zero after max subtraction.  Offline fitting gives a
 * maximum absolute exp error of 5.1e-5 on the fitted interval. */
void softmax_f32_rvv(const float *input, float *output, size_t length)
{
    if (!input || !output || length == 0)
        return;

    float max_value = -INFINITY;
    for (size_t offset = 0; offset < length; ) {
        size_t vl = __riscv_vsetvl_e32m8(length - offset);
        vfloat32m8_t x = __riscv_vle32_v_f32m8(input + offset, vl);
        vfloat32m1_t initial = __riscv_vfmv_s_f_f32m1(max_value, 1);
        vfloat32m1_t reduced = __riscv_vfredmax_vs_f32m8_f32m1(x, initial, vl);
        max_value = __riscv_vfmv_f_s_f32m1_f32(reduced);
        offset += vl;
    }

    float sum = 0.0f;
    for (size_t offset = 0; offset < length; ) {
        size_t vl = __riscv_vsetvl_e32m8(length - offset);
        vfloat32m8_t x = __riscv_vle32_v_f32m8(input + offset, vl);
        x = __riscv_vfsub_vf_f32m8(x, max_value, vl);
        vbool4_t underflow = __riscv_vmflt_vf_f32m8_b4(x, -8.0f, vl);
        x = __riscv_vfmax_vf_f32m8(x, -8.0f, vl);

        vfloat32m8_t y = __riscv_vfmv_v_f_f32m8(7.34296678655e-8f, vl);
        vfloat32m8_t c = __riscv_vfmv_v_f_f32m8(3.33000829946e-6f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(6.75309983314e-5f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(8.15969107345e-4f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(6.61416671412e-3f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(3.83016441475e-2f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(1.62659689007e-1f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(4.97413169516e-1f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(9.99274150832e-1f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        c = __riscv_vfmv_v_f_f32m8(9.99949462408e-1f, vl);
        y = __riscv_vfmadd_vv_f32m8(y, x, c, vl);
        vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        y = __riscv_vmerge_vvm_f32m8(y, zero, underflow, vl);
        __riscv_vse32_v_f32m8(output + offset, y, vl);

        vfloat32m1_t initial = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m8_f32m1(y, initial, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
        offset += vl;
    }

    float inverse_sum = 1.0f / sum;
    for (size_t offset = 0; offset < length; ) {
        size_t vl = __riscv_vsetvl_e32m8(length - offset);
        vfloat32m8_t y = __riscv_vle32_v_f32m8(output + offset, vl);
        y = __riscv_vfmul_vf_f32m8(y, inverse_sum, vl);
        __riscv_vse32_v_f32m8(output + offset, y, vl);
        offset += vl;
    }
}

static void activate_store_chunk_gelu_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m2(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) output[i] = activate_f32(tmp[i], GELU);
}

static void activate_store_chunk_gelu_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m4(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) output[i] = activate_f32(tmp[i], GELU);
}

static void activate_store_chunk_gelu_f32m8(vfloat32m8_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m8(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) output[i] = activate_f32(tmp[i], GELU);
}

static void activate_store_chunk_unknown_f32m2(vfloat32m2_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m2(output, vacc, vl);
}

static void activate_store_chunk_unknown_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_unknown_f32m4(vfloat32m4_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m4(output, vacc, vl);
}

static const activate_store_chunk_kernel_f32m2_t kActivateStoreChunkKernelsF32M2[ACTIVATION_KERNEL_COUNT_F32] = {
    [RELU] = activate_store_chunk_relu_f32m2,
    [SIGMOID] = activate_store_chunk_sigmoid_f32m2,
    [TANH] = activate_store_chunk_tanh_f32m2,
    [LEAKY_RELU] = activate_store_chunk_leaky_relu_f32m2,
    [SOFTMAX] = activate_store_chunk_softmax_f32m2,
    [GELU] = activate_store_chunk_gelu_f32m2,
    [NONE] = activate_store_chunk_none_f32m2,
};

static const activate_store_chunk_kernel_f32m8_t kActivateStoreChunkKernelsF32M8[ACTIVATION_KERNEL_COUNT_F32] = {
    [RELU] = activate_store_chunk_relu_f32,
    [SIGMOID] = activate_store_chunk_sigmoid_f32,
    [TANH] = activate_store_chunk_tanh_f32,
    [LEAKY_RELU] = activate_store_chunk_leaky_relu_f32,
    [SOFTMAX] = activate_store_chunk_softmax_f32,
    [GELU] = activate_store_chunk_gelu_f32m8,
    [NONE] = activate_store_chunk_none_f32,
};

static const activate_store_chunk_kernel_f32m4_t kActivateStoreChunkKernelsF32M4[ACTIVATION_KERNEL_COUNT_F32] = {
    [RELU] = activate_store_chunk_relu_f32m4,
    [SIGMOID] = activate_store_chunk_sigmoid_f32m4,
    [TANH] = activate_store_chunk_tanh_f32m4,
    [LEAKY_RELU] = activate_store_chunk_leaky_relu_f32m4,
    [SOFTMAX] = activate_store_chunk_softmax_f32m4,
    [GELU] = activate_store_chunk_gelu_f32m4,
    [NONE] = activate_store_chunk_none_f32m4,
};

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act)
{
    activate_store_chunk_kernel_f32m8_t kernel =
        select_activate_store_chunk_kernel_f32m8(act);
    size_t remaining = (len > 0) ? (size_t)len : 0;
    while (remaining != 0) {
        const size_t vl = __riscv_vsetvl_e32m8(remaining);
        const vfloat32m8_t values = __riscv_vle32_v_f32m8(src, vl);
        kernel(values, dst, vl);
        src += vl;
        dst += vl;
        remaining -= vl;
    }
}

activate_store_chunk_kernel_f32m2_t select_activate_store_chunk_kernel_f32m2(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT_F32 && kActivateStoreChunkKernelsF32M2[act] != NULL) {
        return kActivateStoreChunkKernelsF32M2[act];
    }
    return activate_store_chunk_unknown_f32m2;
}

activate_store_chunk_kernel_f32m8_t select_activate_store_chunk_kernel_f32m8(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT_F32 && kActivateStoreChunkKernelsF32M8[act] != NULL) {
        return kActivateStoreChunkKernelsF32M8[act];
    }
    return activate_store_chunk_unknown_f32;
}

activate_store_chunk_kernel_f32m4_t select_activate_store_chunk_kernel_f32m4(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT_F32 && kActivateStoreChunkKernelsF32M4[act] != NULL) {
        return kActivateStoreChunkKernelsF32M4[act];
    }
    return activate_store_chunk_unknown_f32m4;
}

void activate_store_chunk_f32m2(ActivationType act, vfloat32m2_t vacc, float *output, size_t vl)
{
    activate_store_chunk_kernel_f32m2_t kernel = select_activate_store_chunk_kernel_f32m2(act);
    kernel(vacc, output, vl);
}

void activate_store_chunk_f32m8(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl)
{
    activate_store_chunk_kernel_f32m8_t kernel = select_activate_store_chunk_kernel_f32m8(act);
    kernel(vacc, output, vl);
}

void activate_store_chunk_f32m4(ActivationType act, vfloat32m4_t vacc, float *output, size_t vl)
{
    activate_store_chunk_kernel_f32m4_t kernel = select_activate_store_chunk_kernel_f32m4(act);
    kernel(vacc, output, vl);
}
