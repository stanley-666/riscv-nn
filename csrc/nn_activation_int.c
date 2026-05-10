#include "nn_activation_int.h"
#include "nn_utils.h"

#include <stdio.h>
#include <stdlib.h>

// Requantize + activation kernels for post-processing int32 accumulators into int8 outputs.

enum { ACTIVATION_KERNEL_COUNT = 7 };

static inline vint8m2_t requantize_vacc_i8_asym_per_channel(vint32m8_t vacc,
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

static void requantize_store_chunk_i8_asym_per_channel_none(vint32m8_t vacc,
                                                            const float *scale,
                                                            const int32_t *zp,
                                                            int8_t *dst,
                                                            size_t vl)
{
    vint8m2_t vq8 = requantize_vacc_i8_asym_per_channel(vacc, scale, zp, vl);
    __riscv_vse8_v_i8m2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_relu(vint32m8_t vacc,
                                                            const float *scale,
                                                            const int32_t *zp,
                                                            int8_t *dst,
                                                            size_t vl)
{
    vint8m2_t vq8 = requantize_vacc_i8_asym_per_channel(vacc, scale, zp, vl);
    vq8 = __riscv_vmax_vx_i8m2(vq8, 0, vl);
    __riscv_vse8_v_i8m2(dst, vq8, vl);
}

static void requantize_store_chunk_i8_asym_per_channel_scalar_with_act(vint32m8_t vacc,
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

static void requantize_store_chunk_i8_asym_per_channel_tanh(vint32m8_t vacc,
                                                            const float *scale,
                                                            const int32_t *zp,
                                                            int8_t *dst,
                                                            size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act(vacc, scale, zp, dst, vl, TANH);
}

static void requantize_store_chunk_i8_asym_per_channel_sigmoid(vint32m8_t vacc,
                                                               const float *scale,
                                                               const int32_t *zp,
                                                               int8_t *dst,
                                                               size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act(vacc, scale, zp, dst, vl, SIGMOID);
}

static void requantize_store_chunk_i8_asym_per_channel_leaky_relu(vint32m8_t vacc,
                                                                  const float *scale,
                                                                  const int32_t *zp,
                                                                  int8_t *dst,
                                                                  size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act(vacc, scale, zp, dst, vl, LEAKY_RELU);
}

static void requantize_store_chunk_i8_asym_per_channel_gelu(vint32m8_t vacc,
                                                            const float *scale,
                                                            const int32_t *zp,
                                                            int8_t *dst,
                                                            size_t vl)
{
    requantize_store_chunk_i8_asym_per_channel_scalar_with_act(vacc, scale, zp, dst, vl, GELU);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_softmax(vint32m8_t vacc,
                                                                           const float *scale,
                                                                           const int32_t *zp,
                                                                           int8_t *dst,
                                                                           size_t vl)
{
    (void)vacc;
    (void)scale;
    (void)zp;
    (void)dst;
    (void)vl;
    printf("SOFTMAX activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_chunk_i8_asym_per_channel_unsupported_unknown(vint32m8_t vacc,
                                                                           const float *scale,
                                                                           const int32_t *zp,
                                                                           int8_t *dst,
                                                                           size_t vl)
{
    (void)vacc;
    (void)scale;
    (void)zp;
    (void)dst;
    (void)vl;
    printf("Unknown activation type in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static const requantize_store_chunk_i8_asym_per_channel_kernel_t
    kRequantizeStoreChunkI8AsymPerChannelKernels[ACTIVATION_KERNEL_COUNT] = {
    [RELU] = requantize_store_chunk_i8_asym_per_channel_relu,
    [SIGMOID] = requantize_store_chunk_i8_asym_per_channel_sigmoid,
    [TANH] = requantize_store_chunk_i8_asym_per_channel_tanh,
    [LEAKY_RELU] = requantize_store_chunk_i8_asym_per_channel_leaky_relu,
    [SOFTMAX] = requantize_store_chunk_i8_asym_per_channel_unsupported_softmax,
    [GELU] = requantize_store_chunk_i8_asym_per_channel_gelu,
    [NONE] = requantize_store_chunk_i8_asym_per_channel_none,
};

requantize_store_chunk_i8_asym_per_channel_kernel_t
select_requantize_store_chunk_i8_asym_per_channel_kernel(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT &&
        kRequantizeStoreChunkI8AsymPerChannelKernels[act] != NULL) {
        return kRequantizeStoreChunkI8AsymPerChannelKernels[act];
    }
    return requantize_store_chunk_i8_asym_per_channel_unsupported_unknown;
}
