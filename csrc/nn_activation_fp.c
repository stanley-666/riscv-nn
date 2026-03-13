#include "nn_activation_fp.h"
#include "nn_utils.h"


void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act)
{
    if (act == RELU) {
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t vdata = __riscv_vle32_v_f32m8(&src[i], vl);
            vdata = __riscv_vfmax_vf_f32m8(vdata, 0.0f, vl);
            __riscv_vse32_v_f32m8(&dst[i], vdata, vl);
            i += vl;
        }
        return;
    } else if (act == SOFTMAX) {
        softmax_f32(src, dst, len);
        return;
    } else if (act == NONE) {
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t vdata = __riscv_vle32_v_f32m8(&src[i], vl);
            __riscv_vse32_v_f32m8(&dst[i], vdata, vl);
            i += vl;
        }
        return;
    } else if (act == LEAKY_RELU) {
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m4(len - i);
            vfloat32m4_t vdata = __riscv_vle32_v_f32m4(&src[i], vl);
            vfloat32m4_t vslope = __riscv_vfmv_v_f_f32m4(0.01f, vl);
            vfloat32m4_t vscaled = __riscv_vfmul_vv_f32m4(vdata, vslope, vl);
            vbool8_t mask = __riscv_vmflt_vf_f32m4_b8(vdata, 0.0f, vl);
            vdata = __riscv_vmerge_vvm_f32m4(vdata, vscaled, mask, vl);
            __riscv_vse32_v_f32m4(&dst[i], vdata, vl);
            i += vl;
        }
        return;
    }

    for (int i = 0; i < len; ++i) {
        dst[i] = activate_f32(src[i], act);
    }
}


void activate_store_chunk_f32(ActivationType act, vfloat32m8_t vacc, float *output, size_t vl)
{
    switch (act) {
    case RELU:
        vacc = __riscv_vfmax_vf_f32m8(vacc, 0.0f, vl);
        __riscv_vse32_v_f32m8(output, vacc, vl);
        return;

    case NONE:
        __riscv_vse32_v_f32m8(output, vacc, vl);
        return;

    case SOFTMAX: 
        __riscv_vse32_v_f32m8(output, vacc, vl);
        return;
        
    case LEAKY_RELU: {
        vfloat32m8_t vslope = __riscv_vfmv_v_f_f32m8(0.01f, vl);
        vfloat32m8_t vscaled = __riscv_vfmul_vv_f32m8(vacc, vslope, vl);
        vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(vacc, 0.0f, vl);
        vacc = __riscv_vmerge_vvm_f32m8(vacc, vscaled, mask, vl);
        __riscv_vse32_v_f32m8(output, vacc, vl);
        return;
    }

    default: {
        float tmp[vl];
        __riscv_vse32_v_f32m8(tmp, vacc, vl);
        for (size_t i = 0; i < vl; ++i) {
            output[i] = activate_f32(tmp[i], act);
        }
        return;
    }
    }
}

