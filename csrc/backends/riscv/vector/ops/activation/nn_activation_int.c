#include "nn_activation_int_internal.h"
#include "nn_utils.h"

#include <stdio.h>
#include <stdlib.h>

enum { ACTIVATION_KERNEL_COUNT = NONE + 1 };

static inline vint8m2_t requantize_vacc_i8_asym_per_channel_m8(vint32m8_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               size_t vl)
{
    vfloat32m8_t vacc_f = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
    vfloat32m8_t vscale = __riscv_vle32_v_f32m8(scale, vl);
    vint32m8_t vzp = __riscv_vle32_v_i32m8(zp, vl);
    vacc_f = __riscv_vfmul_vv_f32m8(vacc_f, vscale, vl);
    vint32m8_t vround = __riscv_vfcvt_x_f_v_i32m8(vacc_f, vl);

    vround = __riscv_vadd_vv_i32m8(vround, vzp, vl);
    vround = __riscv_vmax_vx_i32m8(vround, INT8_MIN, vl);
    vround = __riscv_vmin_vx_i32m8(vround, INT8_MAX, vl);

    vint16m4_t vq16 = __riscv_vncvt_x_x_w_i16m4(vround, vl);
    return __riscv_vncvt_x_x_w_i8m2(vq16, vl);
}

static inline vint8m1_t requantize_vacc_i8_asym_per_channel_m4(vint32m4_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               size_t vl)
{
    vfloat32m4_t vacc_f = __riscv_vfcvt_f_x_v_f32m4(vacc, vl);
    vfloat32m4_t vscale = __riscv_vle32_v_f32m4(scale, vl);
    vint32m4_t vzp = __riscv_vle32_v_i32m4(zp, vl);
    vacc_f = __riscv_vfmul_vv_f32m4(vacc_f, vscale, vl);
    vint32m4_t vround = __riscv_vfcvt_x_f_v_i32m4(vacc_f, vl);

    vround = __riscv_vadd_vv_i32m4(vround, vzp, vl);
    vround = __riscv_vmax_vx_i32m4(vround, INT8_MIN, vl);
    vround = __riscv_vmin_vx_i32m4(vround, INT8_MAX, vl);

    vint16m2_t vq16 = __riscv_vncvt_x_x_w_i16m2(vround, vl);
    return __riscv_vncvt_x_x_w_i8m1(vq16, vl);
}

static inline vint8mf2_t requantize_vacc_i8_asym_per_channel_m2(vint32m2_t vacc,
                                                                const float *scale,
                                                                const int32_t *zp,
                                                                size_t vl)
{
    vfloat32m2_t vacc_f = __riscv_vfcvt_f_x_v_f32m2(vacc, vl);
    vfloat32m2_t vscale = __riscv_vle32_v_f32m2(scale, vl);
    vint32m2_t vzp = __riscv_vle32_v_i32m2(zp, vl);
    vacc_f = __riscv_vfmul_vv_f32m2(vacc_f, vscale, vl);
    vint32m2_t vround = __riscv_vfcvt_x_f_v_i32m2(vacc_f, vl);

    vround = __riscv_vadd_vv_i32m2(vround, vzp, vl);
    vround = __riscv_vmax_vx_i32m2(vround, INT8_MIN, vl);
    vround = __riscv_vmin_vx_i32m2(vround, INT8_MAX, vl);

    vint16m1_t vq16 = __riscv_vncvt_x_x_w_i16m1(vround, vl);
    return __riscv_vncvt_x_x_w_i8mf2(vq16, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_none_m8(vint32m8_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8m2_t vq8 = requantize_vacc_i8_asym_per_channel_m8(vacc, scale, zp, vl);
    __riscv_vse8_v_i8m2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_none_m4(vint32m4_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8m1_t vq8 = requantize_vacc_i8_asym_per_channel_m4(vacc, scale, zp, vl);
    __riscv_vse8_v_i8m1(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_none_m2(vint32m2_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8mf2_t vq8 = requantize_vacc_i8_asym_per_channel_m2(vacc, scale, zp, vl);
    __riscv_vse8_v_i8mf2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_relu_m8(vint32m8_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8m2_t vq8 = requantize_vacc_i8_asym_per_channel_m8(vacc, scale, zp, vl);
    vq8 = __riscv_vmax_vx_i8m2(vq8, 0, vl);
    __riscv_vse8_v_i8m2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_relu_m4(vint32m4_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8m1_t vq8 = requantize_vacc_i8_asym_per_channel_m4(vacc, scale, zp, vl);
    vq8 = __riscv_vmax_vx_i8m1(vq8, 0, vl);
    __riscv_vse8_v_i8m1(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_relu_m2(vint32m2_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    vint8mf2_t vq8 = requantize_vacc_i8_asym_per_channel_m2(vacc, scale, zp, vl);
    vq8 = __riscv_vmax_vx_i8mf2(vq8, 0, vl);
    __riscv_vse8_v_i8mf2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m8(vint32m8_t vacc,
                                                                          const float *scale,
                                                                          const int32_t *zp,
                                                                          int8_t *dst,
                                                                          size_t vl,
                                                                          ActivationType act)
{
    int32_t tmp[vl];
    __riscv_vse32_v_i32m8(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        int8_t rq = requantize_int8_asymmetric(tmp[i], scale[i], zp[i]);
        dst[i] = activate_i8(rq, act);
    }
}

static void requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m4(vint32m4_t vacc,
                                                                          const float *scale,
                                                                          const int32_t *zp,
                                                                          int8_t *dst,
                                                                          size_t vl,
                                                                          ActivationType act)
{
    int32_t tmp[vl];
    __riscv_vse32_v_i32m4(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        int8_t rq = requantize_int8_asymmetric(tmp[i], scale[i], zp[i]);
        dst[i] = activate_i8(rq, act);
    }
}

static void requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m2(vint32m2_t vacc,
                                                                          const float *scale,
                                                                          const int32_t *zp,
                                                                          int8_t *dst,
                                                                          size_t vl,
                                                                          ActivationType act)
{
    int32_t tmp[vl];
    __riscv_vse32_v_i32m2(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        int8_t rq = requantize_int8_asymmetric(tmp[i], scale[i], zp[i]);
        dst[i] = activate_i8(rq, act);
    }
}

static void requantize_store_chunk_i8_asym_per_channel_tanh_m8(vint32m8_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m8(vacc, scale, zp, dst, vl, TANH);
}

static void requantize_store_chunk_i8_asym_per_channel_tanh_m4(vint32m4_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m4(vacc, scale, zp, dst, vl, TANH);
}

static void requantize_store_chunk_i8_asym_per_channel_tanh_m2(vint32m2_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m2(vacc, scale, zp, dst, vl, TANH);
}

static void requantize_store_chunk_i8_asym_per_channel_sigmoid_m8(vint32m8_t vacc,
                                                                  const float *scale,
                                                                  const int32_t *zp,
                                                                  int8_t *dst,
                                                                  size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m8(vacc, scale, zp, dst, vl, SIGMOID);
}

static void requantize_store_chunk_i8_asym_per_channel_sigmoid_m4(vint32m4_t vacc,
                                                                  const float *scale,
                                                                  const int32_t *zp,
                                                                  int8_t *dst,
                                                                  size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m4(vacc, scale, zp, dst, vl, SIGMOID);
}

static void requantize_store_chunk_i8_asym_per_channel_sigmoid_m2(vint32m2_t vacc,
                                                                  const float *scale,
                                                                  const int32_t *zp,
                                                                  int8_t *dst,
                                                                  size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m2(vacc, scale, zp, dst, vl, SIGMOID);
}

static void requantize_store_chunk_i8_asym_per_channel_leaky_relu_m8(vint32m8_t vacc,
                                                                     const float *scale,
                                                                     const int32_t *zp,
                                                                     int8_t *dst,
                                                                     size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m8(vacc, scale, zp, dst, vl, LEAKY_RELU);
}

static void requantize_store_chunk_i8_asym_per_channel_leaky_relu_m4(vint32m4_t vacc,
                                                                     const float *scale,
                                                                     const int32_t *zp,
                                                                     int8_t *dst,
                                                                     size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m4(vacc, scale, zp, dst, vl, LEAKY_RELU);
}

static void requantize_store_chunk_i8_asym_per_channel_leaky_relu_m2(vint32m2_t vacc,
                                                                     const float *scale,
                                                                     const int32_t *zp,
                                                                     int8_t *dst,
                                                                     size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m2(vacc, scale, zp, dst, vl, LEAKY_RELU);
}

static void requantize_store_chunk_i8_asym_per_channel_gelu_m8(vint32m8_t vacc, const float *scale, const int32_t *zp, int8_t *dst, size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m8(vacc, scale, zp, dst, vl, GELU);
}

static void requantize_store_chunk_i8_asym_per_channel_gelu_m4(vint32m4_t vacc, const float *scale, const int32_t *zp, int8_t *dst, size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m4(vacc, scale, zp, dst, vl, GELU);
}

static void requantize_store_chunk_i8_asym_per_channel_gelu_m2(vint32m2_t vacc, const float *scale, const int32_t *zp, int8_t *dst, size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act_m2(vacc, scale, zp, dst, vl, GELU);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m8(vint32m8_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("SOFTMAX activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m4(vint32m4_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("SOFTMAX activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m2(vint32m2_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("SOFTMAX activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m8(vint32m8_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("Unknown activation type in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m4(vint32m4_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("Unknown activation type in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m2(vint32m2_t vacc,
                                                                              const float *scale,
                                                                              const int32_t *zp,
                                                                              int8_t *dst,
                                                                              size_t vl)
{
    (void)vacc; (void)scale; (void)zp; (void)dst; (void)vl;
    printf("Unknown activation type in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static const requantize_store_chunk_i8_asym_per_channel_kernel_m8_t
    kRequantizeStoreChunkI8AsymPerChannelKernelsM8[ACTIVATION_KERNEL_COUNT] = {
    [RELU] = requantize_store_chunk_i8_asym_per_channel_relu_m8,
    [SIGMOID] = requantize_store_chunk_i8_asym_per_channel_sigmoid_m8,
    [TANH] = requantize_store_chunk_i8_asym_per_channel_tanh_m8,
    [LEAKY_RELU] = requantize_store_chunk_i8_asym_per_channel_leaky_relu_m8,
    [SOFTMAX] = requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m8,
    [GELU] = requantize_store_chunk_i8_asym_per_channel_gelu_m8,
    [NONE] = requantize_store_chunk_i8_asym_per_channel_none_m8,
};

static const requantize_store_chunk_i8_asym_per_channel_kernel_m4_t
    kRequantizeStoreChunkI8AsymPerChannelKernelsM4[ACTIVATION_KERNEL_COUNT] = {
    [RELU] = requantize_store_chunk_i8_asym_per_channel_relu_m4,
    [SIGMOID] = requantize_store_chunk_i8_asym_per_channel_sigmoid_m4,
    [TANH] = requantize_store_chunk_i8_asym_per_channel_tanh_m4,
    [LEAKY_RELU] = requantize_store_chunk_i8_asym_per_channel_leaky_relu_m4,
    [SOFTMAX] = requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m4,
    [GELU] = requantize_store_chunk_i8_asym_per_channel_gelu_m4,
    [NONE] = requantize_store_chunk_i8_asym_per_channel_none_m4,
};

static const requantize_store_chunk_i8_asym_per_channel_kernel_m2_t
    kRequantizeStoreChunkI8AsymPerChannelKernelsM2[ACTIVATION_KERNEL_COUNT] = {
    [RELU] = requantize_store_chunk_i8_asym_per_channel_relu_m2,
    [SIGMOID] = requantize_store_chunk_i8_asym_per_channel_sigmoid_m2,
    [TANH] = requantize_store_chunk_i8_asym_per_channel_tanh_m2,
    [LEAKY_RELU] = requantize_store_chunk_i8_asym_per_channel_leaky_relu_m2,
    [SOFTMAX] = requantize_store_chunk_i8_asym_per_channel_unsupported_softmax_m2,
    [GELU] = requantize_store_chunk_i8_asym_per_channel_gelu_m2,
    [NONE] = requantize_store_chunk_i8_asym_per_channel_none_m2,
};

requantize_store_chunk_i8_asym_per_channel_kernel_m8_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT &&
        kRequantizeStoreChunkI8AsymPerChannelKernelsM8[act] != NULL) {
        return kRequantizeStoreChunkI8AsymPerChannelKernelsM8[act];
    }
    return requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m8;
}

requantize_store_chunk_i8_asym_per_channel_kernel_m4_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT &&
        kRequantizeStoreChunkI8AsymPerChannelKernelsM4[act] != NULL) {
        return kRequantizeStoreChunkI8AsymPerChannelKernelsM4[act];
    }
    return requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m4;
}

requantize_store_chunk_i8_asym_per_channel_kernel_m2_t
select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT &&
        kRequantizeStoreChunkI8AsymPerChannelKernelsM2[act] != NULL) {
        return kRequantizeStoreChunkI8AsymPerChannelKernelsM2[act];
    }
    return requantize_store_chunk_i8_asym_per_channel_unsupported_unknown_m2;
}
