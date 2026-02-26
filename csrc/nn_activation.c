#include "nn_activation.h"
#include "nn_utils.h"

#include <stdio.h>
#include <stdlib.h>

enum { ACTIVATION_KERNEL_COUNT = 6 };

static inline vint8m2_t requantize_i8_chunk_rvv(const int32_t *src,
                                                 const float *scale,
                                                 const int32_t *zp,
                                                 int i,
                                                 size_t vl)
{
    vint32m8_t vacc = __riscv_vle32_v_i32m8(&src[i], vl);

    vfloat32m8_t vacc_f = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
    vfloat32m8_t vscale = __riscv_vle32_v_f32m8(&scale[i], vl);
    vacc_f = __riscv_vfmul_vv_f32m8(vacc_f, vscale, vl);

    vint32m8_t vround = __riscv_vfcvt_x_f_v_i32m8(vacc_f, vl);
    vint32m8_t vzp = __riscv_vle32_v_i32m8(&zp[i], vl);
    vround = __riscv_vadd_vv_i32m8(vround, vzp, vl);

    vround = __riscv_vmax_vx_i32m8(vround, INT8_MIN, vl);
    vround = __riscv_vmin_vx_i32m8(vround, INT8_MAX, vl);

    vint16m4_t vq16 = __riscv_vncvt_x_x_w_i16m4(vround, vl);
    return __riscv_vncvt_x_x_w_i8m2(vq16, vl);
}

static void requantize_store_none_rvv(const int32_t *src,
                                      const float *scale,
                                      const int32_t *zp,
                                      int8_t *dst,
                                      int len)
{
    for (int i = 0; i < len; ) {
        size_t vl = __riscv_vsetvl_e32m8(len - i);
        vint8m2_t vq8 = requantize_i8_chunk_rvv(src, scale, zp, i, vl);
        __riscv_vse8_v_i8m2(&dst[i], vq8, vl);
        i += vl;
    }
}

static void requantize_store_relu_rvv(const int32_t *src,
                                      const float *scale,
                                      const int32_t *zp,
                                      int8_t *dst,
                                      int len)
{
    for (int i = 0; i < len; ) {
        size_t vl = __riscv_vsetvl_e32m8(len - i);
        vint8m2_t vq8 = requantize_i8_chunk_rvv(src, scale, zp, i, vl);
        //vint8m2_t vzero = __riscv_vmv_v_x_i8m2(0, vl);
        vq8 = __riscv_vmax_vx_i8m2(vq8, 0, vl);
        __riscv_vse8_v_i8m2(&dst[i], vq8, vl);
        i += vl;
    }
}

static void requantize_store_scalar_with_act(const int32_t *src,
                                             const float *scale,
                                             const int32_t *zp,
                                             int8_t *dst,
                                             int len,
                                             ActivationType act)
{
    for (int i = 0; i < len; ++i) {
        int8_t rq = requantize_int8(src[i], scale[i], zp[i]);
        dst[i] = activate_i8(rq, act);
    }
}

static void requantize_store_tanh_scalar(const int32_t *src,
                                         const float *scale,
                                         const int32_t *zp,
                                         int8_t *dst,
                                         int len)
{
    requantize_store_scalar_with_act(src, scale, zp, dst, len, TANH);
}

static void requantize_store_leaky_relu_scalar(const int32_t *src,
                                               const float *scale,
                                               const int32_t *zp,
                                               int8_t *dst,
                                               int len)
{
    requantize_store_scalar_with_act(src, scale, zp, dst, len, LEAKY_RELU);
}

static void requantize_store_unsupported_sigmoid(const int32_t *src,
                                                 const float *scale,
                                                 const int32_t *zp,
                                                 int8_t *dst,
                                                 int len)
{
    (void)src;
    (void)scale;
    (void)zp;
    (void)dst;
    (void)len;
    printf("SIGMOID activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_unsupported_softmax(const int32_t *src,
                                                 const float *scale,
                                                 const int32_t *zp,
                                                 int8_t *dst,
                                                 int len)
{
    (void)src;
    (void)scale;
    (void)zp;
    (void)dst;
    (void)len;
    printf("SOFTMAX activation not implemented in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static void requantize_store_unsupported_unknown(const int32_t *src,
                                                 const float *scale,
                                                 const int32_t *zp,
                                                 int8_t *dst,
                                                 int len)
{
    (void)src;
    (void)scale;
    (void)zp;
    (void)dst;
    (void)len;
    printf("Unknown activation type in requantize_activate_store_rvv.\n");
    exit(EXIT_FAILURE);
}

static const requantize_store_kernel_t kRequantizeStoreKernels[ACTIVATION_KERNEL_COUNT] = {
    [RELU] = requantize_store_relu_rvv,
    [SIGMOID] = requantize_store_unsupported_sigmoid,
    [TANH] = requantize_store_tanh_scalar,
    [LEAKY_RELU] = requantize_store_leaky_relu_scalar,
    [SOFTMAX] = requantize_store_unsupported_softmax,
    [NONE] = requantize_store_none_rvv,
};

requantize_store_kernel_t select_requantize_store_kernel(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT && kRequantizeStoreKernels[act] != NULL) {
        return kRequantizeStoreKernels[act];
    }
    return requantize_store_unsupported_unknown;
}

void requantize_activate_store_rvv(const int32_t *src,
                                   const float *scale,
                                   const int32_t *zp,
                                   int8_t *dst,
                                   int len,
                                   ActivationType act)
{
    requantize_store_kernel_t kernel = select_requantize_store_kernel(act);
    kernel(src, scale, zp, dst, len);
}
