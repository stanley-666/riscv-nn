#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_fc_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void fullyconnected_int8_vpu_m8(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維

    const int8_t *input_i8 = (const int8_t*)input;
    int8_t *output_i8 = (int8_t*)output;
    const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps; // zero point
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e16m4(outW - o);
        vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            int16_t inval = (int16_t)input_i8[i];
            if (inval == 0) continue;
            const int16_t *wt_ptr = &wt_T[i * outW + o];
            vint16m4_t vwt16  = __riscv_vle16_v_i16m4(wt_ptr, vl);
            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining2_m8(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維
    const int8_t *input_i8 = (const int8_t*)input;
    int8_t *output_i8 = (int8_t*)output;
    const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps; // zero point
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    int32_t *acc_buffer = (int32_t *)layer->params.fc.acc_buffer; // 預先配置的 FC 累加暫存
    if (!acc_buffer)
        acc_buffer = (int32_t *)safe_malloc(outW * sizeof(int32_t));

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e16m4(outW - o);
        vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[o], vl);
        int i = 0;
        for (; i+1 < inDim; i+=2) {
            int16_t inval0 = (int16_t) input_i8[i];
            int16_t inval1 = (int16_t) input_i8[i + 1];
            const int16_t *wt_ptr0 = &wt_T[i * outW + o];
            const int16_t *wt_ptr1 = &wt_T[(i + 1) * outW + o];

            if (inval0 != 0 && inval1 != 0) {     
                vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
            }
            else {
                if (inval0 != 0) {
                    vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                }
                if (inval1 != 0) {
                    vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
                }
            }
        }

        if (i < inDim) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m4_t vwt_tail = __riscv_vle16_v_i16m4(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining4_m8(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維
    ActivationType act = layer->activation;

    const int8_t *input_i8 = (const int8_t*)input;
    int8_t *output_i8 = (int8_t*)output;
    const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps; // zero point
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(act);

    int32_t *acc_buffer = (int32_t *)layer->params.fc.acc_buffer; // 預先配置的 FC 累加暫存
    if (!acc_buffer)
        acc_buffer = (int32_t *)safe_malloc(outW * sizeof(int32_t));

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e16m4(outW - o);
        vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[o], vl);
        int i = 0;
        for (; i+3 < inDim; i+=4) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            const int16_t *wt_ptr0 = &wt_T[i * outW + o];
            const int16_t *wt_ptr1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt_ptr2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt_ptr3 = &wt_T[(i + 3) * outW + o];
            
            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                vint16m4_t vwt16_2  = __riscv_vle16_v_i16m4(wt_ptr2, vl);
                vint16m4_t vwt16_3  = __riscv_vle16_v_i16m4(wt_ptr3, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt16_2, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt16_3, vl);
            }
            else {
                if (inval0 != 0) {
                    vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                }
                if (inval1 != 0) {
                    vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
                }
                if (inval2 != 0) {
                    vint16m4_t vwt16_2  = __riscv_vle16_v_i16m4(wt_ptr2, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt16_2, vl);
                }
                if (inval3 != 0) {
                    vint16m4_t vwt16_3  = __riscv_vle16_v_i16m4(wt_ptr3, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt16_3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m4_t vwt_tail = __riscv_vle16_v_i16m4(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining8_m8(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維
    ActivationType act = layer->activation;

    const int8_t *input_i8 = (const int8_t*)input;
    int8_t *output_i8 = (int8_t*)output;
    const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps; // zero point
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(act);

    int32_t *acc_buffer = (int32_t *)layer->params.fc.acc_buffer; // 預先配置的 FC 累加暫存
    if (!acc_buffer)
        acc_buffer = (int32_t *)safe_malloc(outW * sizeof(int32_t));

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e16m4(outW - o);
        vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[o], vl);
        int i = 0;
        for (; i+7 < inDim; i+=8) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            int16_t inval4 = (int16_t)input_i8[i + 4];
            int16_t inval5 = (int16_t)input_i8[i + 5];
            int16_t inval6 = (int16_t)input_i8[i + 6];
            int16_t inval7 = (int16_t)input_i8[i + 7];

            const int16_t *wt_ptr0 = &wt_T[i * outW + o];
            const int16_t *wt_ptr1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt_ptr2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt_ptr3 = &wt_T[(i + 3) * outW + o];
            const int16_t *wt_ptr4 = &wt_T[(i + 4) * outW + o];
            const int16_t *wt_ptr5 = &wt_T[(i + 5) * outW + o];
            const int16_t *wt_ptr6 = &wt_T[(i + 6) * outW + o];
            const int16_t *wt_ptr7 = &wt_T[(i + 7) * outW + o];

            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0 && inval4 != 0 && inval5 != 0 && inval6 != 0 && inval7 != 0) {
                vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                vint16m4_t vwt16_2  = __riscv_vle16_v_i16m4(wt_ptr2, vl);
                vint16m4_t vwt16_3  = __riscv_vle16_v_i16m4(wt_ptr3, vl);
                vint16m4_t vwt16_4  = __riscv_vle16_v_i16m4(wt_ptr4, vl);
                vint16m4_t vwt16_5  = __riscv_vle16_v_i16m4(wt_ptr5, vl);
                vint16m4_t vwt16_6  = __riscv_vle16_v_i16m4(wt_ptr6, vl);
                vint16m4_t vwt16_7  = __riscv_vle16_v_i16m4(wt_ptr7, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt16_2, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt16_3, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt16_4, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt16_5, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt16_6, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt16_7, vl);
            }
            else {
                if (inval0 != 0) {
                    vint16m4_t vwt16_0  = __riscv_vle16_v_i16m4(wt_ptr0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt16_0, vl);
                }
                if (inval1 != 0) {
                    vint16m4_t vwt16_1  = __riscv_vle16_v_i16m4(wt_ptr1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt16_1, vl);
                }
                if (inval2 != 0) {
                    vint16m4_t vwt16_2  = __riscv_vle16_v_i16m4(wt_ptr2, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt16_2, vl);
                }
                if (inval3 != 0) {
                    vint16m4_t vwt16_3  = __riscv_vle16_v_i16m4(wt_ptr3, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt16_3, vl);
                }
                if (inval4 != 0) {
                    vint16m4_t vwt16_4  = __riscv_vle16_v_i16m4(wt_ptr4, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt16_4, vl);
                }
                if (inval5 != 0) {
                    vint16m4_t vwt16_5  = __riscv_vle16_v_i16m4(wt_ptr5, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt16_5, vl);
                }
                if (inval6 != 0) {
                    vint16m4_t vwt16_6  = __riscv_vle16_v_i16m4(wt_ptr6, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt16_6, vl);
                }
                if (inval7 != 0) {
                    vint16m4_t vwt16_7  = __riscv_vle16_v_i16m4(wt_ptr7, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt16_7, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m4_t vwt_tail = __riscv_vle16_v_i16m4(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m2(outW - o);
        vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            int16_t inval = (int16_t)input_i8[i];
            if (inval == 0) {
                continue;
            }

            const int16_t *wt_ptr = &wt_T[i * outW + o];
            vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt_ptr, vl);
            vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining2_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m2(outW - o);
        vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[o], vl);
        int i = 0;

        for (; i + 1 < inDim; i += 2) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];

            if (inval0 != 0 && inval1 != 0) {
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl);
            } else {
                if (inval0 != 0) {
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0) {
                    vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl);
                }
            }
        }

        if (i < inDim) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m2_t vwt_tail = __riscv_vle16_v_i16m2(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining4_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m2(outW - o);
        vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[o], vl);
        int i = 0;

        for (; i + 3 < inDim; i += 4) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt3 = &wt_T[(i + 3) * outW + o];

            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl);
                vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl);
                vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval2, vwt2, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval3, vwt3, vl);
            } else {
                if (inval0 != 0) {
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0) {
                    vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0) {
                    vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0) {
                    vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval3, vwt3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m2_t vwt_tail = __riscv_vle16_v_i16m2(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining8_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m2(outW - o);
        vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[o], vl);
        int i = 0;

        for (; i + 7 < inDim; i += 8) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            int16_t inval4 = (int16_t)input_i8[i + 4];
            int16_t inval5 = (int16_t)input_i8[i + 5];
            int16_t inval6 = (int16_t)input_i8[i + 6];
            int16_t inval7 = (int16_t)input_i8[i + 7];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt3 = &wt_T[(i + 3) * outW + o];
            const int16_t *wt4 = &wt_T[(i + 4) * outW + o];
            const int16_t *wt5 = &wt_T[(i + 5) * outW + o];
            const int16_t *wt6 = &wt_T[(i + 6) * outW + o];
            const int16_t *wt7 = &wt_T[(i + 7) * outW + o];

            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0 &&
                inval4 != 0 && inval5 != 0 && inval6 != 0 && inval7 != 0) {
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl);
                vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl);
                vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl);
                vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt4, vl);
                vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt5, vl);
                vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt6, vl);
                vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt7, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval2, vwt2, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval3, vwt3, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval4, vwt4, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval5, vwt5, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval6, vwt6, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval7, vwt7, vl);
            } else {
                if (inval0 != 0) { vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval0, vwt0, vl); }
                if (inval1 != 0) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval1, vwt1, vl); }
                if (inval2 != 0) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval2, vwt2, vl); }
                if (inval3 != 0) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval3, vwt3, vl); }
                if (inval4 != 0) { vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt4, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval4, vwt4, vl); }
                if (inval5 != 0) { vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt5, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval5, vwt5, vl); }
                if (inval6 != 0) { vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt6, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval6, vwt6, vl); }
                if (inval7 != 0) { vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt7, vl); vacc = __riscv_vwmacc_vx_i32m4(vacc, inval7, vwt7, vl); }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m2_t vwt_tail = __riscv_vle16_v_i16m2(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m1(outW - o);
        vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            int16_t inval = (int16_t)input_i8[i];
            if (inval == 0) {
                continue;
            }

            const int16_t *wt_ptr = &wt_T[i * outW + o];
            vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt_ptr, vl);
            vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining2_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m1(outW - o);
        vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[o], vl);
        int i = 0;

        for (; i + 1 < inDim; i += 2) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];

            if (inval0 != 0 && inval1 != 0) {
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl);
            } else {
                if (inval0 != 0) {
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0) {
                    vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl);
                }
            }
        }

        if (i < inDim) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m1_t vwt_tail = __riscv_vle16_v_i16m1(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining4_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m1(outW - o);
        vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[o], vl);
        int i = 0;

        for (; i + 3 < inDim; i += 4) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt3 = &wt_T[(i + 3) * outW + o];

            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl);
                vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl);
                vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval2, vwt2, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval3, vwt3, vl);
            } else {
                if (inval0 != 0) {
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0) {
                    vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0) {
                    vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0) {
                    vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval3, vwt3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m1_t vwt_tail = __riscv_vle16_v_i16m1(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_int8_vpu_chaining8_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;

    const int8_t *input_i8 = (const int8_t *)input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.fc.bias;
    const float *M = (const float *)layer->params.fc.M;
    const int32_t *Z = (const int32_t *)layer->params.fc.zps;
    const int16_t *wt_T = (const int16_t *)layer->params.fc.weights_rvv;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e16m1(outW - o);
        vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[o], vl);
        int i = 0;

        for (; i + 7 < inDim; i += 8) {
            int16_t inval0 = (int16_t)input_i8[i];
            int16_t inval1 = (int16_t)input_i8[i + 1];
            int16_t inval2 = (int16_t)input_i8[i + 2];
            int16_t inval3 = (int16_t)input_i8[i + 3];
            int16_t inval4 = (int16_t)input_i8[i + 4];
            int16_t inval5 = (int16_t)input_i8[i + 5];
            int16_t inval6 = (int16_t)input_i8[i + 6];
            int16_t inval7 = (int16_t)input_i8[i + 7];
            const int16_t *wt0 = &wt_T[i * outW + o];
            const int16_t *wt1 = &wt_T[(i + 1) * outW + o];
            const int16_t *wt2 = &wt_T[(i + 2) * outW + o];
            const int16_t *wt3 = &wt_T[(i + 3) * outW + o];
            const int16_t *wt4 = &wt_T[(i + 4) * outW + o];
            const int16_t *wt5 = &wt_T[(i + 5) * outW + o];
            const int16_t *wt6 = &wt_T[(i + 6) * outW + o];
            const int16_t *wt7 = &wt_T[(i + 7) * outW + o];

            if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0 &&
                inval4 != 0 && inval5 != 0 && inval6 != 0 && inval7 != 0) {
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl);
                vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl);
                vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl);
                vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt4, vl);
                vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt5, vl);
                vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt6, vl);
                vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt7, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval2, vwt2, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval3, vwt3, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval4, vwt4, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval5, vwt5, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval6, vwt6, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval7, vwt7, vl);
            } else {
                if (inval0 != 0) { vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval0, vwt0, vl); }
                if (inval1 != 0) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval1, vwt1, vl); }
                if (inval2 != 0) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval2, vwt2, vl); }
                if (inval3 != 0) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval3, vwt3, vl); }
                if (inval4 != 0) { vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt4, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval4, vwt4, vl); }
                if (inval5 != 0) { vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt5, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval5, vwt5, vl); }
                if (inval6 != 0) { vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt6, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval6, vwt6, vl); }
                if (inval7 != 0) { vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt7, vl); vacc = __riscv_vwmacc_vx_i32m2(vacc, inval7, vwt7, vl); }
            }
        }

        for (; i < inDim; ++i) {
            int16_t in_tail = (int16_t)input_i8[i];
            if (in_tail != 0) {
                const int16_t *wt_tail = &wt_T[i * outW + o];
                vint16m1_t vwt_tail = __riscv_vle16_v_i16m1(wt_tail, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        rq_activation_kernel(vacc, &M[o], &Z[o], &output_i8[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_m8(NNModule *layer, void *input, void *output)
{   // self
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維
    ActivationType act = layer->activation;

    const float *input_f32 = (const float*)input;
    float *output_f32 = (float*)output;
    const float *bias_f32 = (const float*)layer->params.fc.bias;

    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m8(outW - o);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            float inval = input_f32[i];
            if (inval == 0.0f) continue;
            const float *wt_ptr = &wt_T[i * outW + o];
            vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt_ptr, vl);
            vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

void fullyconnected_fp32_vpu_unroll2_m8(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W*C), Output: length outW; both contiguous.
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m8(outW - o);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[o], vl);

        int i = 0;
        for (; i + 1 < inDim; i += 2) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];

            if(inval0 != 0.0f && inval1 != 0.0f) {
                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                }

                if (inval1 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                }
            }
        }
        if (i < inDim) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m8_t vwt_tail = __riscv_vle32_v_f32m8(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll4_m8(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m8(outW - o);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[o], vl);
        int i = 0;
        for (; i + 3 < inDim; i += 4) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];

            if( inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f) {   
                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
            }
            else {
                if (inval0 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0.0f) {
                    vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0.0f) {
                    vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m8_t vwt_tail = __riscv_vle32_v_f32m8(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll8_m8(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m8(outW - o);
        vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[o], vl);

        int i = 0;
        for (; i + 7 < inDim; i += 8) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            float inval4 = input_f32[i + 4];
            float inval5 = input_f32[i + 5];
            float inval6 = input_f32[i + 6];
            float inval7 = input_f32[i + 7];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];
            const float *wt4 = &wt_T[(i + 4) * outW + o];
            const float *wt5 = &wt_T[(i + 5) * outW + o];
            const float *wt6 = &wt_T[(i + 6) * outW + o];
            const float *wt7 = &wt_T[(i + 7) * outW + o];

            if( inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f &&
                inval4 != 0.0f && inval5 != 0.0f && inval6 != 0.0f && inval7 != 0.0f) {

                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                vfloat32m8_t vwt4 = __riscv_vle32_v_f32m8(wt4, vl);
                vfloat32m8_t vwt5 = __riscv_vle32_v_f32m8(wt5, vl);
                vfloat32m8_t vwt6 = __riscv_vle32_v_f32m8(wt6, vl);
                vfloat32m8_t vwt7 = __riscv_vle32_v_f32m8(wt7, vl);    

                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval4, vwt4, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval5, vwt5, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval6, vwt6, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval7, vwt7, vl);
            }
            else {
                if (inval0 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0.0f) {
                    vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0.0f) {
                    vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
                }
                if (inval4 != 0.0f) {
                    vfloat32m8_t vwt4 = __riscv_vle32_v_f32m8(wt4, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval4, vwt4, vl);
                }
                if (inval5 != 0.0f) {
                    vfloat32m8_t vwt5 = __riscv_vle32_v_f32m8(wt5, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval5, vwt5, vl);
                }
                if (inval6 != 0.0f) {
                    vfloat32m8_t vwt6 = __riscv_vle32_v_f32m8(wt6, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval6, vwt6, vl);
                }
                if (inval7 != 0.0f) {
                    vfloat32m8_t vwt7 = __riscv_vle32_v_f32m8(wt7, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval7, vwt7, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m8_t vwt_tail = __riscv_vle32_v_f32m8(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m4(outW - o);
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            float inval = input_f32[i];
            if (inval == 0.0f) {
                continue;
            }

            const float *wt_ptr = &wt_T[i * outW + o];
            vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt_ptr, vl);
            vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll2_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m4(outW - o);
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[o], vl);
        int i = 0;

        for (; i + 1 < inDim; i += 2) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f) {
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
                }
            }
        }

        if (i < inDim) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m4_t vwt_tail = __riscv_vle32_v_f32m4(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll4_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m4(outW - o);
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[o], vl);
        int i = 0;

        for (; i + 3 < inDim; i += 4) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f) {
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                vfloat32m4_t vwt2 = __riscv_vle32_v_f32m4(wt2, vl);
                vfloat32m4_t vwt3 = __riscv_vle32_v_f32m4(wt3, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, vwt3, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0.0f) {
                    vfloat32m4_t vwt2 = __riscv_vle32_v_f32m4(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0.0f) {
                    vfloat32m4_t vwt3 = __riscv_vle32_v_f32m4(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, vwt3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m4_t vwt_tail = __riscv_vle32_v_f32m4(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll8_m4(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(act);

    for (int o = 0; o < outW; ) {
        size_t vl = __riscv_vsetvl_e32m4(outW - o);
        vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[o], vl);
        int i = 0;

        for (; i + 7 < inDim; i += 8) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            float inval4 = input_f32[i + 4];
            float inval5 = input_f32[i + 5];
            float inval6 = input_f32[i + 6];
            float inval7 = input_f32[i + 7];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];
            const float *wt4 = &wt_T[(i + 4) * outW + o];
            const float *wt5 = &wt_T[(i + 5) * outW + o];
            const float *wt6 = &wt_T[(i + 6) * outW + o];
            const float *wt7 = &wt_T[(i + 7) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f &&
                inval4 != 0.0f && inval5 != 0.0f && inval6 != 0.0f && inval7 != 0.0f) {
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                vfloat32m4_t vwt2 = __riscv_vle32_v_f32m4(wt2, vl);
                vfloat32m4_t vwt3 = __riscv_vle32_v_f32m4(wt3, vl);
                vfloat32m4_t vwt4 = __riscv_vle32_v_f32m4(wt4, vl);
                vfloat32m4_t vwt5 = __riscv_vle32_v_f32m4(wt5, vl);
                vfloat32m4_t vwt6 = __riscv_vle32_v_f32m4(wt6, vl);
                vfloat32m4_t vwt7 = __riscv_vle32_v_f32m4(wt7, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, vwt3, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval4, vwt4, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval5, vwt5, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval6, vwt6, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval7, vwt7, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0.0f) {
                    vfloat32m4_t vwt2 = __riscv_vle32_v_f32m4(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0.0f) {
                    vfloat32m4_t vwt3 = __riscv_vle32_v_f32m4(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, vwt3, vl);
                }
                if (inval4 != 0.0f) {
                    vfloat32m4_t vwt4 = __riscv_vle32_v_f32m4(wt4, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval4, vwt4, vl);
                }
                if (inval5 != 0.0f) {
                    vfloat32m4_t vwt5 = __riscv_vle32_v_f32m4(wt5, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval5, vwt5, vl);
                }
                if (inval6 != 0.0f) {
                    vfloat32m4_t vwt6 = __riscv_vle32_v_f32m4(wt6, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval6, vwt6, vl);
                }
                if (inval7 != 0.0f) {
                    vfloat32m4_t vwt7 = __riscv_vle32_v_f32m4(wt7, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval7, vwt7, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m4_t vwt_tail = __riscv_vle32_v_f32m4(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(act);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e32m2(outW - o);
        vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[o], vl);

        for (int i = 0; i < inDim; ++i) {
            float inval = input_f32[i];
            if (inval == 0.0f) {
                continue;
            }

            const float *wt_ptr = &wt_T[i * outW + o];
            vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt_ptr, vl);
            vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll2_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(act);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e32m2(outW - o);
        vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[o], vl);
        int i = 0;

        for (; i + 1 < inDim; i += 2) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f) {
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl);
                vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl);
                }
            }
        }

        if (i < inDim) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m2_t vwt_tail = __riscv_vle32_v_f32m2(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll4_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(act);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e32m2(outW - o);
        vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[o], vl);
        int i = 0;

        for (; i + 3 < inDim; i += 4) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f) {
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl);
                vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl);
                vfloat32m2_t vwt2 = __riscv_vle32_v_f32m2(wt2, vl);
                vfloat32m2_t vwt3 = __riscv_vle32_v_f32m2(wt3, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, vwt3, vl);
            } else {
                if (inval0 != 0.0f) {
                    vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl);
                }
                if (inval1 != 0.0f) {
                    vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl);
                }
                if (inval2 != 0.0f) {
                    vfloat32m2_t vwt2 = __riscv_vle32_v_f32m2(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, vwt2, vl);
                }
                if (inval3 != 0.0f) {
                    vfloat32m2_t vwt3 = __riscv_vle32_v_f32m2(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, vwt3, vl);
                }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m2_t vwt_tail = __riscv_vle32_v_f32m2(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}

void fullyconnected_fp32_vpu_unroll8_m2(NNModule *layer, void *input, void *output)
{
    int inDim = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W;
    ActivationType act = layer->activation;

    const float *input_f32 = (const float *)input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.fc.bias;
    const float *wt_T = (const float *)layer->params.fc.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(act);

    for (int o = 0; o < outW;) {
        size_t vl = __riscv_vsetvl_e32m2(outW - o);
        vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[o], vl);
        int i = 0;

        for (; i + 7 < inDim; i += 8) {
            float inval0 = input_f32[i];
            float inval1 = input_f32[i + 1];
            float inval2 = input_f32[i + 2];
            float inval3 = input_f32[i + 3];
            float inval4 = input_f32[i + 4];
            float inval5 = input_f32[i + 5];
            float inval6 = input_f32[i + 6];
            float inval7 = input_f32[i + 7];
            const float *wt0 = &wt_T[i * outW + o];
            const float *wt1 = &wt_T[(i + 1) * outW + o];
            const float *wt2 = &wt_T[(i + 2) * outW + o];
            const float *wt3 = &wt_T[(i + 3) * outW + o];
            const float *wt4 = &wt_T[(i + 4) * outW + o];
            const float *wt5 = &wt_T[(i + 5) * outW + o];
            const float *wt6 = &wt_T[(i + 6) * outW + o];
            const float *wt7 = &wt_T[(i + 7) * outW + o];

            if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f &&
                inval4 != 0.0f && inval5 != 0.0f && inval6 != 0.0f && inval7 != 0.0f) {
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl);
                vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl);
                vfloat32m2_t vwt2 = __riscv_vle32_v_f32m2(wt2, vl);
                vfloat32m2_t vwt3 = __riscv_vle32_v_f32m2(wt3, vl);
                vfloat32m2_t vwt4 = __riscv_vle32_v_f32m2(wt4, vl);
                vfloat32m2_t vwt5 = __riscv_vle32_v_f32m2(wt5, vl);
                vfloat32m2_t vwt6 = __riscv_vle32_v_f32m2(wt6, vl);
                vfloat32m2_t vwt7 = __riscv_vle32_v_f32m2(wt7, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, vwt2, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, vwt3, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval4, vwt4, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval5, vwt5, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval6, vwt6, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval7, vwt7, vl);
            } else {
                if (inval0 != 0.0f) { vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, vwt0, vl); }
                if (inval1 != 0.0f) { vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, vwt1, vl); }
                if (inval2 != 0.0f) { vfloat32m2_t vwt2 = __riscv_vle32_v_f32m2(wt2, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, vwt2, vl); }
                if (inval3 != 0.0f) { vfloat32m2_t vwt3 = __riscv_vle32_v_f32m2(wt3, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, vwt3, vl); }
                if (inval4 != 0.0f) { vfloat32m2_t vwt4 = __riscv_vle32_v_f32m2(wt4, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval4, vwt4, vl); }
                if (inval5 != 0.0f) { vfloat32m2_t vwt5 = __riscv_vle32_v_f32m2(wt5, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval5, vwt5, vl); }
                if (inval6 != 0.0f) { vfloat32m2_t vwt6 = __riscv_vle32_v_f32m2(wt6, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval6, vwt6, vl); }
                if (inval7 != 0.0f) { vfloat32m2_t vwt7 = __riscv_vle32_v_f32m2(wt7, vl); vacc = __riscv_vfmacc_vf_f32m2(vacc, inval7, vwt7, vl); }
            }
        }

        for (; i < inDim; ++i) {
            float in_tail = input_f32[i];
            if (in_tail != 0.0f) {
                const float *wt_tail = &wt_T[i * outW + o];
                vfloat32m2_t vwt_tail = __riscv_vle32_v_f32m2(wt_tail, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, in_tail, vwt_tail, vl);
            }
        }

        activation_kernel(vacc, &output_f32[o], vl);
        o += (int)vl;
    }
}
