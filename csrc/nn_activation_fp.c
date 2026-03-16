#include "nn_activation_fp.h"
#include "nn_utils.h"

enum { ACTIVATION_KERNEL_COUNT_F32 = 6 };

static void activate_store_chunk_none_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_relu_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    vacc = __riscv_vfmax_vf_f32m8(vacc, 0.0f, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_leaky_relu_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    vfloat32m8_t vslope = __riscv_vfmv_v_f_f32m8(0.01f, vl);
    vfloat32m8_t vscaled = __riscv_vfmul_vv_f32m8(vacc, vslope, vl);
    vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(vacc, 0.0f, vl);
    vacc = __riscv_vmerge_vvm_f32m8(vacc, vscaled, mask, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_sigmoid_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    float tmp[vl];
    __riscv_vse32_v_f32m8(tmp, vacc, vl);
    for (size_t i = 0; i < vl; ++i) {
        output[i] = activate_f32(tmp[i], SIGMOID);
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

static void activate_store_chunk_softmax_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}

static void activate_store_chunk_unknown_f32(vfloat32m8_t vacc, float *output, size_t vl)
{
    __riscv_vse32_v_f32m8(output, vacc, vl);
}


static const activate_store_chunk_kernel_f32_t kActivateStoreChunkKernelsF32[ACTIVATION_KERNEL_COUNT_F32] = {
    [RELU] = activate_store_chunk_relu_f32,
    [SIGMOID] = activate_store_chunk_sigmoid_f32,
    [TANH] = activate_store_chunk_tanh_f32,
    [LEAKY_RELU] = activate_store_chunk_leaky_relu_f32,
    [SOFTMAX] = activate_store_chunk_softmax_f32,
    [NONE] = activate_store_chunk_none_f32,
};

activate_store_chunk_kernel_f32_t select_activate_store_chunk_kernel_f32(ActivationType act)
{
    if ((unsigned)act < ACTIVATION_KERNEL_COUNT_F32 && kActivateStoreChunkKernelsF32[act] != NULL) {
        return kActivateStoreChunkKernelsF32[act];
    }
    return activate_store_chunk_unknown_f32;
}

void activate_store_chunk_f32(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl)
{
    activate_store_chunk_kernel_f32_t kernel = select_activate_store_chunk_kernel_f32(act);
    kernel(vacc, output, vl);
}
