#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_conv1d_fp32_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif
#include <time.h>

void conv1d_fp32_vpu_m8(NNModule *layer, void *input, void *output)
{
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const float *input_f32   = (const float *)padded_input;
    float       *output_f32  = (float *)output;
    const float *bias_f32    = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    /* Reorder kernel to (K * inC, outC) layout for contiguous vector loads */

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {   
                    if(in_ptr[ic] == 0.0f) continue;
                    float inval = in_ptr[ic];
                    int col_idx = k * inC + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_fp32_vpu_nobranch_m8(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_nobranch.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const float *input_f32   = (const float *)padded_input;
    float       *output_f32  = (float *)output;
    const float *bias_f32    = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    int col_idx = k * inC + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_fp32_vpu_im2col_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

/*
 * Register-file pressure estimate for FP32 weight-reuse kernels.
 *
 * Main live vector groups in the MAC loop are approximately:
 *   reuse_w_N: N accumulator groups + 1 weight group.
 *
 * Estimated architectural vector registers:
 *   reuse_w_2_m8: (2 + 1) * LMUL8 = 24 regs, within 32-reg VRF.
 *   reuse_w_4_m8: (4 + 1) * LMUL8 = 40 regs, exceeds 32-reg VRF.
 *   reuse_w_8_m8: (8 + 1) * LMUL8 = 72 regs, exceeds 32-reg VRF.
 *
 * This excludes activation temporaries and compiler allocation details, so it
 * is a lower-bound design estimate rather than proof of spilling.
 */
void conv1d_fp32_vpu_im2col_reuse_w_2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 4) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc3 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval3, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 8) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];
        const float *row4 = &input_f32[(pos + 4) * cols];
        const float *row5 = &input_f32[(pos + 5) * cols];
        const float *row6 = &input_f32[(pos + 6) * cols];
        const float *row7 = &input_f32[(pos + 7) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc3 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc4 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc5 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc6 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc7 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                float inval4 = row4[col_idx];
                float inval5 = row5[col_idx];
                float inval6 = row6[col_idx];
                float inval7 = row7[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f &&
                    inval4 == 0.0f && inval5 == 0.0f &&
                    inval6 == 0.0f && inval7 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval3, vwt, vl);
                }
                if (inval4 != 0.0f) {
                    vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval4, vwt, vl);
                }
                if (inval5 != 0.0f) {
                    vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval5, vwt, vl);
                }
                if (inval6 != 0.0f) {
                    vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval6, vwt, vl);
                }
                if (inval7 != 0.0f) {
                    vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval7, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &output_f32[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &output_f32[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &output_f32[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &output_f32[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval00 = row0[col_idx + 0];
                float inval01 = row0[col_idx + 1];
                float inval10 = row1[col_idx + 0];
                float inval11 = row1[col_idx + 1];

                if (inval00 == 0.0f && inval01 == 0.0f &&
                    inval10 == 0.0f && inval11 == 0.0f) {
                    continue;
                }

                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                if (inval00 != 0.0f || inval10 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    if (inval00 != 0.0f) {
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval00, vwt0, vl);
                    }
                    if (inval10 != 0.0f) {
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval10, vwt0, vl);
                    }
                }
                if (inval01 != 0.0f || inval11 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    if (inval01 != 0.0f) {
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval01, vwt1, vl);
                    }
                    if (inval11 != 0.0f) {
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval11, vwt1, vl);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval00 = row0[col_idx + 0];
                float inval01 = row0[col_idx + 1];
                float inval02 = row0[col_idx + 2];
                float inval03 = row0[col_idx + 3];
                float inval10 = row1[col_idx + 0];
                float inval11 = row1[col_idx + 1];
                float inval12 = row1[col_idx + 2];
                float inval13 = row1[col_idx + 3];

                if (inval00 == 0.0f && inval01 == 0.0f &&
                    inval02 == 0.0f && inval03 == 0.0f &&
                    inval10 == 0.0f && inval11 == 0.0f &&
                    inval12 == 0.0f && inval13 == 0.0f) {
                    continue;
                }

                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                if (inval00 != 0.0f || inval10 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    if (inval00 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval00, vwt0, vl);
                    if (inval10 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval10, vwt0, vl);
                }
                if (inval01 != 0.0f || inval11 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    if (inval01 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval01, vwt1, vl);
                    if (inval11 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval11, vwt1, vl);
                }
                if (inval02 != 0.0f || inval12 != 0.0f) {
                    vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                    if (inval02 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval02, vwt2, vl);
                    if (inval12 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval12, vwt2, vl);
                }
                if (inval03 != 0.0f || inval13 != 0.0f) {
                    vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                    if (inval03 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval03, vwt3, vl);
                    if (inval13 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval13, vwt3, vl);
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval00 = row0[col_idx + 0];
                float inval01 = row0[col_idx + 1];
                float inval02 = row0[col_idx + 2];
                float inval03 = row0[col_idx + 3];
                float inval04 = row0[col_idx + 4];
                float inval05 = row0[col_idx + 5];
                float inval06 = row0[col_idx + 6];
                float inval07 = row0[col_idx + 7];
                float inval10 = row1[col_idx + 0];
                float inval11 = row1[col_idx + 1];
                float inval12 = row1[col_idx + 2];
                float inval13 = row1[col_idx + 3];
                float inval14 = row1[col_idx + 4];
                float inval15 = row1[col_idx + 5];
                float inval16 = row1[col_idx + 6];
                float inval17 = row1[col_idx + 7];

                if (inval00 == 0.0f && inval01 == 0.0f &&
                    inval02 == 0.0f && inval03 == 0.0f &&
                    inval04 == 0.0f && inval05 == 0.0f &&
                    inval06 == 0.0f && inval07 == 0.0f &&
                    inval10 == 0.0f && inval11 == 0.0f &&
                    inval12 == 0.0f && inval13 == 0.0f &&
                    inval14 == 0.0f && inval15 == 0.0f &&
                    inval16 == 0.0f && inval17 == 0.0f) {
                    continue;
                }

                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const float *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const float *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const float *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const float *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];
                if (inval00 != 0.0f || inval10 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    if (inval00 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval00, vwt0, vl);
                    if (inval10 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval10, vwt0, vl);
                }
                if (inval01 != 0.0f || inval11 != 0.0f) {
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    if (inval01 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval01, vwt1, vl);
                    if (inval11 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval11, vwt1, vl);
                }
                if (inval02 != 0.0f || inval12 != 0.0f) {
                    vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                    if (inval02 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval02, vwt2, vl);
                    if (inval12 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval12, vwt2, vl);
                }
                if (inval03 != 0.0f || inval13 != 0.0f) {
                    vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                    if (inval03 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval03, vwt3, vl);
                    if (inval13 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval13, vwt3, vl);
                }
                if (inval04 != 0.0f || inval14 != 0.0f) {
                    vfloat32m8_t vwt4 = __riscv_vle32_v_f32m8(wt4, vl);
                    if (inval04 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval04, vwt4, vl);
                    if (inval14 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval14, vwt4, vl);
                }
                if (inval05 != 0.0f || inval15 != 0.0f) {
                    vfloat32m8_t vwt5 = __riscv_vle32_v_f32m8(wt5, vl);
                    if (inval05 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval05, vwt5, vl);
                    if (inval15 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval15, vwt5, vl);
                }
                if (inval06 != 0.0f || inval16 != 0.0f) {
                    vfloat32m8_t vwt6 = __riscv_vle32_v_f32m8(wt6, vl);
                    if (inval06 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval06, vwt6, vl);
                    if (inval16 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval16, vwt6, vl);
                }
                if (inval07 != 0.0f || inval17 != 0.0f) {
                    vfloat32m8_t vwt7 = __riscv_vle32_v_f32m8(wt7, vl);
                    if (inval07 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval07, vwt7, vl);
                    if (inval17 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval17, vwt7, vl);
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, vwt, vl);
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m8_t vacc00 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc01 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc10 = vacc00;
            vfloat32m8_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval00 = row0[col_idx + 0];
                float inval01 = row0[col_idx + 1];
                float inval10 = row1[col_idx + 0];
                float inval11 = row1[col_idx + 1];
                if (inval00 == 0.0f && inval01 == 0.0f &&
                    inval10 == 0.0f && inval11 == 0.0f) {
                    continue;
                }

                const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                if (inval00 != 0.0f || inval10 != 0.0f) {
                    vfloat32m8_t vwt00 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                    if (inval00 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, inval00, vwt00, vls[0]);
                    if (inval10 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, inval10, vwt00, vls[0]);
                }
                if (inval01 != 0.0f || inval11 != 0.0f) {
                    vfloat32m8_t vwt01 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                    if (inval01 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, inval01, vwt01, vls[0]);
                    if (inval11 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, inval11, vwt01, vls[0]);
                }
                if (blocks > 1) {
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval00 != 0.0f || inval10 != 0.0f) {
                        vfloat32m8_t vwt10 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        if (inval00 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, inval00, vwt10, vls[1]);
                        if (inval10 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, inval10, vwt10, vls[1]);
                    }
                    if (inval01 != 0.0f || inval11 != 0.0f) {
                        vfloat32m8_t vwt11 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        if (inval01 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, inval01, vwt11, vls[1]);
                        if (inval11 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, inval11, vwt11, vls[1]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m8_t vacc00 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc01 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc10 = vacc00;
            vfloat32m8_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float in00 = row0[col_idx + 0];
                float in01 = row0[col_idx + 1];
                float in02 = row0[col_idx + 2];
                float in03 = row0[col_idx + 3];
                float in10 = row1[col_idx + 0];
                float in11 = row1[col_idx + 1];
                float in12 = row1[col_idx + 2];
                float in13 = row1[col_idx + 3];
                if (in00 == 0.0f && in01 == 0.0f && in02 == 0.0f && in03 == 0.0f &&
                    in10 == 0.0f && in11 == 0.0f && in12 == 0.0f && in13 == 0.0f) {
                    continue;
                }

                const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                if (in00 != 0.0f || in10 != 0.0f) {
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt00, vls[0]);
                    if (in00 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in00, vwt, vls[0]);
                    if (in10 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in10, vwt, vls[0]);
                }
                if (in01 != 0.0f || in11 != 0.0f) {
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt01, vls[0]);
                    if (in01 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in01, vwt, vls[0]);
                    if (in11 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in11, vwt, vls[0]);
                }
                if (in02 != 0.0f || in12 != 0.0f) {
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt02, vls[0]);
                    if (in02 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in02, vwt, vls[0]);
                    if (in12 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in12, vwt, vls[0]);
                }
                if (in03 != 0.0f || in13 != 0.0f) {
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt03, vls[0]);
                    if (in03 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in03, vwt, vls[0]);
                    if (in13 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in13, vwt, vls[0]);
                }

                if (blocks > 1) {
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (in00 != 0.0f || in10 != 0.0f) {
                        vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        if (in00 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in00, vwt, vls[1]);
                        if (in10 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in10, vwt, vls[1]);
                    }
                    if (in01 != 0.0f || in11 != 0.0f) {
                        vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        if (in01 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in01, vwt, vls[1]);
                        if (in11 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in11, vwt, vls[1]);
                    }
                    if (in02 != 0.0f || in12 != 0.0f) {
                        vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt12, vls[1]);
                        if (in02 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in02, vwt, vls[1]);
                        if (in12 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in12, vwt, vls[1]);
                    }
                    if (in03 != 0.0f || in13 != 0.0f) {
                        vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt13, vls[1]);
                        if (in03 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in03, vwt, vls[1]);
                        if (in13 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in13, vwt, vls[1]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m8_t vacc00 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc01 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc10 = vacc00;
            vfloat32m8_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float in00 = row0[col_idx + 0], in01 = row0[col_idx + 1];
                float in02 = row0[col_idx + 2], in03 = row0[col_idx + 3];
                float in04 = row0[col_idx + 4], in05 = row0[col_idx + 5];
                float in06 = row0[col_idx + 6], in07 = row0[col_idx + 7];
                float in10 = row1[col_idx + 0], in11 = row1[col_idx + 1];
                float in12 = row1[col_idx + 2], in13 = row1[col_idx + 3];
                float in14 = row1[col_idx + 4], in15 = row1[col_idx + 5];
                float in16 = row1[col_idx + 6], in17 = row1[col_idx + 7];
                if (in00 == 0.0f && in01 == 0.0f && in02 == 0.0f && in03 == 0.0f &&
                    in04 == 0.0f && in05 == 0.0f && in06 == 0.0f && in07 == 0.0f &&
                    in10 == 0.0f && in11 == 0.0f && in12 == 0.0f && in13 == 0.0f &&
                    in14 == 0.0f && in15 == 0.0f && in16 == 0.0f && in17 == 0.0f) {
                    continue;
                }

                const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                const float *wt04 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                const float *wt05 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                const float *wt06 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                const float *wt07 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                if (in00 != 0.0f || in10 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt00, vls[0]); if (in00 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in00, v, vls[0]); if (in10 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in10, v, vls[0]); }
                if (in01 != 0.0f || in11 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt01, vls[0]); if (in01 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in01, v, vls[0]); if (in11 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in11, v, vls[0]); }
                if (in02 != 0.0f || in12 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt02, vls[0]); if (in02 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in02, v, vls[0]); if (in12 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in12, v, vls[0]); }
                if (in03 != 0.0f || in13 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt03, vls[0]); if (in03 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in03, v, vls[0]); if (in13 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in13, v, vls[0]); }
                if (in04 != 0.0f || in14 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt04, vls[0]); if (in04 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in04, v, vls[0]); if (in14 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in14, v, vls[0]); }
                if (in05 != 0.0f || in15 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt05, vls[0]); if (in05 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in05, v, vls[0]); if (in15 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in15, v, vls[0]); }
                if (in06 != 0.0f || in16 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt06, vls[0]); if (in06 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in06, v, vls[0]); if (in16 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in16, v, vls[0]); }
                if (in07 != 0.0f || in17 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt07, vls[0]); if (in07 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, in07, v, vls[0]); if (in17 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, in17, v, vls[0]); }

                if (blocks > 1) {
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt14 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const float *wt15 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const float *wt16 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const float *wt17 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (in00 != 0.0f || in10 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt10, vls[1]); if (in00 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in00, v, vls[1]); if (in10 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in10, v, vls[1]); }
                    if (in01 != 0.0f || in11 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt11, vls[1]); if (in01 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in01, v, vls[1]); if (in11 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in11, v, vls[1]); }
                    if (in02 != 0.0f || in12 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt12, vls[1]); if (in02 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in02, v, vls[1]); if (in12 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in12, v, vls[1]); }
                    if (in03 != 0.0f || in13 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt13, vls[1]); if (in03 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in03, v, vls[1]); if (in13 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in13, v, vls[1]); }
                    if (in04 != 0.0f || in14 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt14, vls[1]); if (in04 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in04, v, vls[1]); if (in14 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in14, v, vls[1]); }
                    if (in05 != 0.0f || in15 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt15, vls[1]); if (in05 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in05, v, vls[1]); if (in15 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in15, v, vls[1]); }
                    if (in06 != 0.0f || in16 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt16, vls[1]); if (in06 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in06, v, vls[1]); if (in16 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in16, v, vls[1]); }
                    if (in07 != 0.0f || in17 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt17, vls[1]); if (in07 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, in07, v, vls[1]); if (in17 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, in17, v, vls[1]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m8(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m8(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m8(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m8(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_m4(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = k * inC + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.
SPDX-License-Identifier: Apache-2.0
Author : Stanley Lee
*/

void conv1d_fp32_vpu_im2col_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
            }

            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

/*
 * Register-file pressure estimate:
 *   reuse_w_2_m4: (2 + 1) * LMUL4 = 12 regs, low pressure.
 *   reuse_w_4_m4: (4 + 1) * LMUL4 = 20 regs, within 32-reg VRF.
 *   reuse_w_8_m4: (8 + 1) * LMUL4 = 36 regs, exceeds 32-reg VRF.
 */
void conv1d_fp32_vpu_im2col_reuse_w_2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *input_row0 = &input_f32[pos * cols];
        const float *input_row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = vacc0;

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = input_row0[col_idx];
                float inval1 = input_row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 4) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval3, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 8) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];
        const float *row4 = &input_f32[(pos + 4) * cols];
        const float *row5 = &input_f32[(pos + 5) * cols];
        const float *row6 = &input_f32[(pos + 6) * cols];
        const float *row7 = &input_f32[(pos + 7) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc4 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc5 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc6 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc7 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                float inval4 = row4[col_idx];
                float inval5 = row5[col_idx];
                float inval6 = row6[col_idx];
                float inval7 = row7[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f &&
                    inval4 == 0.0f && inval5 == 0.0f &&
                    inval6 == 0.0f && inval7 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval3, vwt, vl);
                }
                if (inval4 != 0.0f) {
                    vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval4, vwt, vl);
                }
                if (inval5 != 0.0f) {
                    vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval5, vwt, vl);
                }
                if (inval6 != 0.0f) {
                    vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval6, vwt, vl);
                }
                if (inval7 != 0.0f) {
                    vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval7, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &output_f32[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &output_f32[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &output_f32[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &output_f32[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                for (int u = 0; u < 2; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                for (int u = 0; u < 4; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                for (int u = 0; u < 8; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m4_t vacc00 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc01 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc10 = vacc00;
            vfloat32m4_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                for (int u = 0; u < 2; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m4_t vacc00 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc01 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc10 = vacc00;
            vfloat32m4_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                for (int u = 0; u < 4; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m4_t vacc00 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc01 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc10 = vacc00;
            vfloat32m4_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                for (int u = 0; u < 8; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m4_t vwt0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m4(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m4(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t vwt1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m4(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m4(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_mask_u2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_mask_u2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    long weight_stride_bytes = (long)outC * (long)sizeof(float);

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    ActivationType act = layer->activation;

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ++oc) {
            float sum = bias_f32[oc];
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                size_t vl = __riscv_vsetvl_e32m1(2);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
            }
            for (; col_idx < cols; ++col_idx) {
                size_t vl = __riscv_vsetvl_e32m1(1);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
            }
            output_f32[pos * outC + oc] = activate_f32(sum, act);
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_mask_u4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_mask_u4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    long weight_stride_bytes = (long)outC * (long)sizeof(float);

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    ActivationType act = layer->activation;

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ++oc) {
            float sum = bias_f32[oc];
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                size_t vl = __riscv_vsetvl_e32m1(4);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
            }
            for (; col_idx < cols; ) {
                size_t vl = __riscv_vsetvl_e32m1(cols - col_idx);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
                col_idx += (int)vl;
            }
            output_f32[pos * outC + oc] = activate_f32(sum, act);
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_mask_u8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_mask_u8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    long weight_stride_bytes = (long)outC * (long)sizeof(float);

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    ActivationType act = layer->activation;

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ++oc) {
            float sum = bias_f32[oc];
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                size_t vl = __riscv_vsetvl_e32m1(8);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
            }
            for (; col_idx < cols; ) {
                size_t vl = __riscv_vsetvl_e32m1(cols - col_idx);
                vfloat32m1_t vin = __riscv_vle32_v_f32m1(&input_row[col_idx], vl);
                vbool32_t mask = __riscv_vmfne_vf_f32m1_b32(vin, 0.0f, vl);
                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                vin = __riscv_vmerge_vvm_f32m1(vzero, vin, mask, vl);
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m1_t vwt = __riscv_vlse32_v_f32m1(wt, weight_stride_bytes, vl);
                vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vin, vwt, vl);
                vfloat32m1_t vred0 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m1_f32m1(vprod, vred0, vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(vred);
                col_idx += (int)vl;
            }
            output_f32[pos * outC + oc] = activate_f32(sum, act);
        }
    }

    safe_free(im2col_input);
}

static void conv1d_fp32_vpu_im2col_nobranch_generic(NNModule *layer,
                                                    void *input,
                                                    void *output,
                                                    int col_unroll,
                                                    int blocks_limit)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_nobranch_generic.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < blocks_limit) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc1 = vacc0;
            vfloat32m8_t vacc2 = vacc0;
            vfloat32m8_t vacc3 = vacc0;
            vfloat32m8_t vacc4 = vacc0;
            vfloat32m8_t vacc5 = vacc0;
            vfloat32m8_t vacc6 = vacc0;
            vfloat32m8_t vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m8(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m8(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m8(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m8(&bias_f32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + col_unroll - 1 < cols; col_idx += col_unroll) {
                for (int u = 0; u < col_unroll; ++u) {
                    float inval = input_row[col_idx + u];
                    if (blocks > 0) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[0]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                    }
                    if (blocks > 1) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v, vls[1]);
                    }
                    if (blocks > 2) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[2]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v, vls[2]);
                    }
                    if (blocks > 3) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[3]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v, vls[3]);
                    }
                    if (blocks > 4) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[4]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval, v, vls[4]);
                    }
                    if (blocks > 5) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[5]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval, v, vls[5]);
                    }
                    if (blocks > 6) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[6]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval, v, vls[6]);
                    }
                    if (blocks > 7) {
                        const float *wt = &weight_buffer[(col_idx + u) * outC + bases[7]];
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval, v, vls[7]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[1]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[2]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[3]);
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[4]);
                    vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[5]);
                    vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[6]);
                    vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[7]);
                    vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 1, 1);
}

void conv1d_fp32_vpu_im2col_unroll2_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 2, 1);
}

void conv1d_fp32_vpu_im2col_unroll2_acc2_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 2, 2);
}

void conv1d_fp32_vpu_im2col_unroll2_acc4_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 2, 4);
}

void conv1d_fp32_vpu_im2col_unroll2_acc8_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 2, 8);
}

void conv1d_fp32_vpu_im2col_unroll4_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 4, 1);
}

void conv1d_fp32_vpu_im2col_unroll4_acc2_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 4, 2);
}

void conv1d_fp32_vpu_im2col_unroll4_acc4_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 4, 4);
}

void conv1d_fp32_vpu_im2col_unroll4_acc8_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 4, 8);
}

void conv1d_fp32_vpu_im2col_unroll8_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 8, 1);
}

void conv1d_fp32_vpu_im2col_unroll8_acc2_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 8, 2);
}

void conv1d_fp32_vpu_im2col_unroll8_acc4_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 8, 4);
}

void conv1d_fp32_vpu_im2col_unroll8_acc8_nobranch_m8(NNModule *layer, void *input, void *output)
{
    conv1d_fp32_vpu_im2col_nobranch_generic(layer, input, output, 8, 8);
}

void conv1d_fp32_vpu_im2col_unroll2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                if (inval0 != 0.0f && inval1 != 0.0f) {
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
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            switch (blocks) {
            case 1: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v1, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
                vfloat32m8_t vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt20, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt21, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval1, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt2, vls[2]);
                    vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc8");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc1 = vacc0;
            vfloat32m8_t vacc2 = vacc0;
            vfloat32m8_t vacc3 = vacc0;
            vfloat32m8_t vacc4 = vacc0;
            vfloat32m8_t vacc5 = vacc0;
            vfloat32m8_t vacc6 = vacc0;
            vfloat32m8_t vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m8(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m8(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m8(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m8(&bias_f32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v, vls[0]);
                    }
                }
                if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v, vls[1]);
                    }
                }
                if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v, vls[2]);
                    }
                }
                if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval0, v, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval1, v, vls[3]);
                    }
                }
                if (blocks > 4) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval0, v, vls[4]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval1, v, vls[4]);
                    }
                }
                if (blocks > 5) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval0, v, vls[5]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval1, v, vls[5]);
                    }
                }
                if (blocks > 6) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval0, v, vls[6]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval1, v, vls[6]);
                    }
                }
                if (blocks > 7) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval0, v, vls[7]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval1, v, vls[7]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[1]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[2]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[3]);
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[4]);
                    vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[5]);
                    vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[6]);
                    vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[7]);
                    vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f) {
                    vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                    vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                    vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                    vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
                } else {
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt0, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, v, vl);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt1, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, v, vl);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt2, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, v, vl);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt3, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, v, vl);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt02, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt03, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            switch (blocks) {
            case 1: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v1, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v2, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v3, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt02, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt03, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v2, vls[2]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt02, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt12, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt22, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval2, v2, vls[2]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt03, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt13, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt23, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval3, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                vfloat32m8_t vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
                vfloat32m8_t vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const float *wt32 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const float *wt33 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt20, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt21, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval1, v3, vls[3]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt02, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt12, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt22, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt32, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval2, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval2, v3, vls[3]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt03, vls[0]);
                        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt13, vls[1]);
                        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt23, vls[2]);
                        vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt33, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval3, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval3, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vfloat32m8_t v2 = __riscv_vle32_v_f32m8(wt2, vls[2]);
                    vfloat32m8_t v3 = __riscv_vle32_v_f32m8(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc8_m8(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0, vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m8(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m8(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m8(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m8(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval0,v,vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval1,v,vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval2,v,vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval3,v,vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval0,v,vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval1,v,vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval2,v,vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval3,v,vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval0,v,vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval1,v,vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval2,v,vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval3,v,vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval0,v,vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval1,v,vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval2,v,vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval3,v,vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m8(vacc4,inval0,v,vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m8(vacc4,inval1,v,vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m8(vacc4,inval2,v,vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m8(vacc4,inval3,v,vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m8(vacc5,inval0,v,vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m8(vacc5,inval1,v,vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m8(vacc5,inval2,v,vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m8(vacc5,inval3,v,vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m8(vacc6,inval0,v,vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m8(vacc6,inval1,v,vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m8(vacc6,inval2,v,vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m8(vacc6,inval3,v,vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m8(vacc7,inval0,v,vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m8(vacc7,inval1,v,vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m8(vacc7,inval2,v,vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m8(vacc7,inval3,v,vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval,v,vls[0]);
                    } else if (b == 1) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval,v,vls[1]);
                    } else if (b == 2) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval,v,vls[2]);
                    } else if (b == 3) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval,v,vls[3]);
                    } else if (b == 4) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[4]);
                        vacc4=__riscv_vfmacc_vf_f32m8(vacc4,inval,v,vls[4]);
                    } else if (b == 5) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[5]);
                        vacc5=__riscv_vfmacc_vf_f32m8(vacc5,inval,v,vls[5]);
                    } else if (b == 6) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[6]);
                        vacc6=__riscv_vfmacc_vf_f32m8(vacc6,inval,v,vls[6]);
                    } else {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[7]);
                        vacc7=__riscv_vfmacc_vf_f32m8(vacc7,inval,v,vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_m8(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4];
                float inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6];
                float inval7 = input_row[col_idx + 7];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const float *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const float *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const float *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const float *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];
                if (inval0 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, v, vl);
                } if (inval1 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, v, vl);
                } if (inval2 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, v, vl);
                } if (inval3 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, v, vl);
                } if (inval4 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt4, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval4, v, vl);
                } if (inval5 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt5, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval5, v, vl);
                } if (inval6 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt6, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval6, v, vl);
                } if (inval7 != 0.0f) {
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt7, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval7, v, vl);
                } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
            } act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc2_m8(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc2");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {
                0};
            size_t vls[2] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt0, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt1, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt2, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt3, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt4, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt5, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt6, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0.0f) { vfloat32m8_t v = __riscv_vle32_v_f32m8(wt7, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval7, v, vls[0]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
                vfloat32m8_t vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt04 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt05 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt06 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt07 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt14 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const float *wt15 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const float *wt16 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const float *wt17 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt00, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt10, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v1, vls[1]); }
                    if (inval1 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt01, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt11, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v1, vls[1]); }
                    if (inval2 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt02, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt12, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v1, vls[1]); }
                    if (inval3 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt03, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt13, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v1, vls[1]); }
                    if (inval4 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt04, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt14, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval4, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval4, v1, vls[1]); }
                    if (inval5 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt05, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt15, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval5, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval5, v1, vls[1]); }
                    if (inval6 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt06, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt16, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval6, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval6, v1, vls[1]); }
                    if (inval7 != 0.0f) { vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt07, vls[0]); vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt17, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval7, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval7, v1, vls[1]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m8_t v0 = __riscv_vle32_v_f32m8(wt0, vls[0]);
                    vfloat32m8_t v1 = __riscv_vle32_v_f32m8(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc4_m8(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc4");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {
                0};
            size_t vls[4] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3], inval4 = input_row[col_idx+4], inval5 = input_row[col_idx+5], inval6 = input_row[col_idx+6], inval7 = input_row[col_idx+7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[0]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[0]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[0]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[0]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[0]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[0]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[0]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval0,v,vls[0]);
                    } if (inval1 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval1,v,vls[0]);
                    } if (inval2 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval2,v,vls[0]);
                    } if (inval3 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval3,v,vls[0]);
                    } if (inval4 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w4,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval4,v,vls[0]);
                    } if (inval5 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w5,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval5,v,vls[0]);
                    } if (inval6 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w6,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval6,v,vls[0]);
                    } if (inval7 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w7,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval7,v,vls[0]);
                    } } if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[1]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[1]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[1]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[1]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[1]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[1]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[1]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval0,v,vls[1]);
                    } if (inval1 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval1,v,vls[1]);
                    } if (inval2 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval2,v,vls[1]);
                    } if (inval3 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval3,v,vls[1]);
                    } if (inval4 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w4,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval4,v,vls[1]);
                    } if (inval5 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w5,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval5,v,vls[1]);
                    } if (inval6 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w6,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval6,v,vls[1]);
                    } if (inval7 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w7,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval7,v,vls[1]);
                    } } if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[2]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[2]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[2]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[2]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[2]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[2]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[2]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval0,v,vls[2]);
                    } if (inval1 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval1,v,vls[2]);
                    } if (inval2 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval2,v,vls[2]);
                    } if (inval3 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval3,v,vls[2]);
                    } if (inval4 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w4,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval4,v,vls[2]);
                    } if (inval5 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w5,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval5,v,vls[2]);
                    } if (inval6 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w6,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval6,v,vls[2]);
                    } if (inval7 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w7,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval7,v,vls[2]);
                    } } if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[3]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[3]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[3]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[3]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[3]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[3]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[3]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w0,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval0,v,vls[3]);
                    } if (inval1 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w1,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval1,v,vls[3]);
                    } if (inval2 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w2,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval2,v,vls[3]);
                    } if (inval3 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w3,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval3,v,vls[3]);
                    } if (inval4 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w4,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval4,v,vls[3]);
                    } if (inval5 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w5,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval5,v,vls[3]);
                    } if (inval6 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w6,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval6,v,vls[3]);
                    } if (inval7 != 0.0f) {
                        vfloat32m8_t v=__riscv_vle32_v_f32m8(w7,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval7,v,vls[3]);
                    } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[0]];
                    vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[0]);
                    vacc0=__riscv_vfmacc_vf_f32m8(vacc0,inval,v,vls[0]);
                } if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[1]];
                    vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[1]);
                    vacc1=__riscv_vfmacc_vf_f32m8(vacc1,inval,v,vls[1]);
                } if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[2]];
                    vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[2]);
                    vacc2=__riscv_vfmacc_vf_f32m8(vacc2,inval,v,vls[2]);
                } if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[3]];
                    vfloat32m8_t v=__riscv_vle32_v_f32m8(wt,vls[3]);
                    vacc3=__riscv_vfmacc_vf_f32m8(vacc3,inval,v,vls[3]);
                } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc8_m8(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t act_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m8(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m8_t vacc0 = __riscv_vle32_v_f32m8(&bias_f32[bases[0]], vls[0]);
            vfloat32m8_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            vfloat32m8_t vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m8(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m8(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m8(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m8(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m8(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m8(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m8(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx + 2) * outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx + 3) * outC + bases[b]];
                    const float *w4 = &weight_buffer[(col_idx + 4) * outC + bases[b]];
                    const float *w5 = &weight_buffer[(col_idx + 5) * outC + bases[b]];
                    const float *w6 = &weight_buffer[(col_idx + 6) * outC + bases[b]];
                    const float *w7 = &weight_buffer[(col_idx + 7) * outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval0, v, vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval1, v, vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval2, v, vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval3, v, vls[0]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval4, v, vls[0]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval5, v, vls[0]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval6, v, vls[0]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval7, v, vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval0, v, vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval1, v, vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval2, v, vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval3, v, vls[1]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval4, v, vls[1]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval5, v, vls[1]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval6, v, vls[1]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval7, v, vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval0, v, vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval1, v, vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval2, v, vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval3, v, vls[2]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval4, v, vls[2]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval5, v, vls[2]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval6, v, vls[2]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval7, v, vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval0, v, vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval1, v, vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval2, v, vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval3, v, vls[3]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval4, v, vls[3]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval5, v, vls[3]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval6, v, vls[3]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval7, v, vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval0, v, vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval1, v, vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval2, v, vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval3, v, vls[4]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval4, v, vls[4]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval5, v, vls[4]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval6, v, vls[4]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval7, v, vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval0, v, vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval1, v, vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval2, v, vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval3, v, vls[5]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval4, v, vls[5]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval5, v, vls[5]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval6, v, vls[5]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval7, v, vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval0, v, vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval1, v, vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval2, v, vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval3, v, vls[6]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval4, v, vls[6]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval5, v, vls[6]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval6, v, vls[6]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval7, v, vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w0, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval0, v, vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w1, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval1, v, vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w2, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval2, v, vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w3, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval3, v, vls[7]);
                        } if (inval4 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w4, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval4, v, vls[7]);
                        } if (inval5 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w5, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval5, v, vls[7]);
                        } if (inval6 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w6, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval6, v, vls[7]);
                        } if (inval7 != 0.0f) {
                            vfloat32m8_t v = __riscv_vle32_v_f32m8(w7, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval7, v, vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m8(vacc0, inval, v, vls[0]);
                    } else if (b == 1) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m8(vacc1, inval, v, vls[1]);
                    } else if (b == 2) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m8(vacc2, inval, v, vls[2]);
                    } else if (b == 3) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m8(vacc3, inval, v, vls[3]);
                    } else if (b == 4) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m8(vacc4, inval, v, vls[4]);
                    } else if (b == 5) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m8(vacc5, inval, v, vls[5]);
                    } else if (b == 6) {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m8(vacc6, inval, v, vls[6]);
                    } else {
                        vfloat32m8_t v = __riscv_vle32_v_f32m8(wt, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m8(vacc7, inval, v, vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
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
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation); 

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            switch (blocks) {
            case 1: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v1, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
                vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt20, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt21, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval1, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt2, vls[2]);
                    vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc8");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc1 = vacc0;
            vfloat32m4_t vacc2 = vacc0;
            vfloat32m4_t vacc3 = vacc0;
            vfloat32m4_t vacc4 = vacc0;
            vfloat32m4_t vacc5 = vacc0;
            vfloat32m4_t vacc6 = vacc0;
            vfloat32m4_t vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m4(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m4(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m4(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m4(&bias_f32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v, vls[0]);
                    }
                }
                if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v, vls[1]);
                    }
                }
                if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v, vls[2]);
                    }
                }
                if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval0, v, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval1, v, vls[3]);
                    }
                }
                if (blocks > 4) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval0, v, vls[4]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval1, v, vls[4]);
                    }
                }
                if (blocks > 5) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval0, v, vls[5]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval1, v, vls[5]);
                    }
                }
                if (blocks > 6) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval0, v, vls[6]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval1, v, vls[6]);
                    }
                }
                if (blocks > 7) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval0, v, vls[7]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval1, v, vls[7]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[1]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[2]);
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[3]);
                    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[4]);
                    vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[5]);
                    vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[6]);
                    vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[7]);
                    vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
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
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, v, vl);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, v, vl);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt2, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, v, vl);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt3, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, v, vl);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt02, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt03, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            switch (blocks) {
            case 1: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v1, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v2, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v3, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt02, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt03, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v2, vls[2]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt02, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt12, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt22, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval2, v2, vls[2]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt03, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt13, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt23, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval3, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                vfloat32m4_t vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
                vfloat32m4_t vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const float *wt32 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const float *wt33 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt20, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt21, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval1, v3, vls[3]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt02, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt12, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt22, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt32, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval2, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval2, v3, vls[3]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt03, vls[0]);
                        vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt13, vls[1]);
                        vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt23, vls[2]);
                        vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt33, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval3, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval3, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vfloat32m4_t v2 = __riscv_vle32_v_f32m4(wt2, vls[2]);
                    vfloat32m4_t v3 = __riscv_vle32_v_f32m4(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc8_m4(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0, vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m4(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m4(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m4(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m4(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval0,v,vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval1,v,vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval2,v,vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval3,v,vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval0,v,vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval1,v,vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval2,v,vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval3,v,vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval0,v,vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval1,v,vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval2,v,vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval3,v,vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval0,v,vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval1,v,vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval2,v,vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval3,v,vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m4(vacc4,inval0,v,vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m4(vacc4,inval1,v,vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m4(vacc4,inval2,v,vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m4(vacc4,inval3,v,vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m4(vacc5,inval0,v,vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m4(vacc5,inval1,v,vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m4(vacc5,inval2,v,vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m4(vacc5,inval3,v,vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m4(vacc6,inval0,v,vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m4(vacc6,inval1,v,vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m4(vacc6,inval2,v,vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m4(vacc6,inval3,v,vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m4(vacc7,inval0,v,vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m4(vacc7,inval1,v,vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m4(vacc7,inval2,v,vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m4(vacc7,inval3,v,vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval,v,vls[0]);
                    } else if (b == 1) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval,v,vls[1]);
                    } else if (b == 2) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval,v,vls[2]);
                    } else if (b == 3) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval,v,vls[3]);
                    } else if (b == 4) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[4]);
                        vacc4=__riscv_vfmacc_vf_f32m4(vacc4,inval,v,vls[4]);
                    } else if (b == 5) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[5]);
                        vacc5=__riscv_vfmacc_vf_f32m4(vacc5,inval,v,vls[5]);
                    } else if (b == 6) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[6]);
                        vacc6=__riscv_vfmacc_vf_f32m4(vacc6,inval,v,vls[6]);
                    } else {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[7]);
                        vacc7=__riscv_vfmacc_vf_f32m4(vacc7,inval,v,vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_m4(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4];
                float inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6];
                float inval7 = input_row[col_idx + 7];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const float *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const float *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const float *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const float *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];
                if (inval0 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval0, v, vl);
                } if (inval1 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval1, v, vl);
                } if (inval2 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval2, v, vl);
                } if (inval3 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval3, v, vl);
                } if (inval4 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt4, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval4, v, vl);
                } if (inval5 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt5, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval5, v, vl);
                } if (inval6 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt6, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval6, v, vl);
                } if (inval7 != 0.0f) {
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt7, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval7, v, vl);
                } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
            } act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc2_m4(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc2");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {
                0};
            size_t vls[2] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt0, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt1, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt2, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt3, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt4, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt5, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt6, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0.0f) { vfloat32m4_t v = __riscv_vle32_v_f32m4(wt7, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval7, v, vls[0]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
                vfloat32m4_t vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt04 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt05 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt06 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt07 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt14 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const float *wt15 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const float *wt16 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const float *wt17 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt00, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt10, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v1, vls[1]); }
                    if (inval1 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt01, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt11, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v1, vls[1]); }
                    if (inval2 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt02, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt12, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v1, vls[1]); }
                    if (inval3 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt03, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt13, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v1, vls[1]); }
                    if (inval4 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt04, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt14, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval4, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval4, v1, vls[1]); }
                    if (inval5 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt05, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt15, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval5, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval5, v1, vls[1]); }
                    if (inval6 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt06, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt16, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval6, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval6, v1, vls[1]); }
                    if (inval7 != 0.0f) { vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt07, vls[0]); vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt17, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval7, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval7, v1, vls[1]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m4_t v0 = __riscv_vle32_v_f32m4(wt0, vls[0]);
                    vfloat32m4_t v1 = __riscv_vle32_v_f32m4(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc4_m4(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc4");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {
                0};
            size_t vls[4] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3], inval4 = input_row[col_idx+4], inval5 = input_row[col_idx+5], inval6 = input_row[col_idx+6], inval7 = input_row[col_idx+7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[0]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[0]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[0]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[0]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[0]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[0]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[0]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval0,v,vls[0]);
                    } if (inval1 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval1,v,vls[0]);
                    } if (inval2 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval2,v,vls[0]);
                    } if (inval3 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval3,v,vls[0]);
                    } if (inval4 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w4,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval4,v,vls[0]);
                    } if (inval5 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w5,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval5,v,vls[0]);
                    } if (inval6 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w6,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval6,v,vls[0]);
                    } if (inval7 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w7,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval7,v,vls[0]);
                    } } if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[1]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[1]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[1]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[1]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[1]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[1]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[1]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval0,v,vls[1]);
                    } if (inval1 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval1,v,vls[1]);
                    } if (inval2 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval2,v,vls[1]);
                    } if (inval3 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval3,v,vls[1]);
                    } if (inval4 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w4,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval4,v,vls[1]);
                    } if (inval5 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w5,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval5,v,vls[1]);
                    } if (inval6 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w6,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval6,v,vls[1]);
                    } if (inval7 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w7,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval7,v,vls[1]);
                    } } if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[2]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[2]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[2]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[2]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[2]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[2]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[2]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval0,v,vls[2]);
                    } if (inval1 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval1,v,vls[2]);
                    } if (inval2 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval2,v,vls[2]);
                    } if (inval3 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval3,v,vls[2]);
                    } if (inval4 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w4,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval4,v,vls[2]);
                    } if (inval5 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w5,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval5,v,vls[2]);
                    } if (inval6 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w6,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval6,v,vls[2]);
                    } if (inval7 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w7,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval7,v,vls[2]);
                    } } if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[3]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[3]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[3]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[3]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[3]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[3]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[3]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w0,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval0,v,vls[3]);
                    } if (inval1 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w1,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval1,v,vls[3]);
                    } if (inval2 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w2,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval2,v,vls[3]);
                    } if (inval3 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w3,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval3,v,vls[3]);
                    } if (inval4 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w4,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval4,v,vls[3]);
                    } if (inval5 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w5,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval5,v,vls[3]);
                    } if (inval6 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w6,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval6,v,vls[3]);
                    } if (inval7 != 0.0f) {
                        vfloat32m4_t v=__riscv_vle32_v_f32m4(w7,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval7,v,vls[3]);
                    } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[0]];
                    vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[0]);
                    vacc0=__riscv_vfmacc_vf_f32m4(vacc0,inval,v,vls[0]);
                } if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[1]];
                    vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[1]);
                    vacc1=__riscv_vfmacc_vf_f32m4(vacc1,inval,v,vls[1]);
                } if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[2]];
                    vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[2]);
                    vacc2=__riscv_vfmacc_vf_f32m4(vacc2,inval,v,vls[2]);
                } if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[3]];
                    vfloat32m4_t v=__riscv_vle32_v_f32m4(wt,vls[3]);
                    vacc3=__riscv_vfmacc_vf_f32m4(vacc3,inval,v,vls[3]);
                } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc8_m4(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t act_kernel = select_activate_store_chunk_kernel_f32m4(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(&bias_f32[bases[0]], vls[0]);
            vfloat32m4_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            vfloat32m4_t vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m4(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m4(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m4(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m4(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m4(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m4(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m4(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx + 2) * outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx + 3) * outC + bases[b]];
                    const float *w4 = &weight_buffer[(col_idx + 4) * outC + bases[b]];
                    const float *w5 = &weight_buffer[(col_idx + 5) * outC + bases[b]];
                    const float *w6 = &weight_buffer[(col_idx + 6) * outC + bases[b]];
                    const float *w7 = &weight_buffer[(col_idx + 7) * outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval0, v, vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval1, v, vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval2, v, vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval3, v, vls[0]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval4, v, vls[0]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval5, v, vls[0]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval6, v, vls[0]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval7, v, vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval0, v, vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval1, v, vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval2, v, vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval3, v, vls[1]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval4, v, vls[1]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval5, v, vls[1]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval6, v, vls[1]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval7, v, vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval0, v, vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval1, v, vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval2, v, vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval3, v, vls[2]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval4, v, vls[2]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval5, v, vls[2]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval6, v, vls[2]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval7, v, vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval0, v, vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval1, v, vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval2, v, vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval3, v, vls[3]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval4, v, vls[3]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval5, v, vls[3]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval6, v, vls[3]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval7, v, vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval0, v, vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval1, v, vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval2, v, vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval3, v, vls[4]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval4, v, vls[4]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval5, v, vls[4]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval6, v, vls[4]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval7, v, vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval0, v, vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval1, v, vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval2, v, vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval3, v, vls[5]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval4, v, vls[5]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval5, v, vls[5]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval6, v, vls[5]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval7, v, vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval0, v, vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval1, v, vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval2, v, vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval3, v, vls[6]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval4, v, vls[6]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval5, v, vls[6]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval6, v, vls[6]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval7, v, vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w0, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval0, v, vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w1, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval1, v, vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w2, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval2, v, vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w3, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval3, v, vls[7]);
                        } if (inval4 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w4, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval4, v, vls[7]);
                        } if (inval5 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w5, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval5, v, vls[7]);
                        } if (inval6 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w6, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval6, v, vls[7]);
                        } if (inval7 != 0.0f) {
                            vfloat32m4_t v = __riscv_vle32_v_f32m4(w7, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval7, v, vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, inval, v, vls[0]);
                    } else if (b == 1) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, inval, v, vls[1]);
                    } else if (b == 2) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, inval, v, vls[2]);
                    } else if (b == 3) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, inval, v, vls[3]);
                    } else if (b == 4) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, inval, v, vls[4]);
                    } else if (b == 5) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, inval, v, vls[5]);
                    } else if (b == 6) {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, inval, v, vls[6]);
                    } else {
                        vfloat32m4_t v = __riscv_vle32_v_f32m4(wt, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m4(vacc7, inval, v, vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}


void conv1d_fp32_vpu_chaining2_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const float *input_f32   = (const float *)padded_input;
    float       *output_f32  = (float *)output;
    const float *bias_f32    = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    float inval0 = in_ptr[ic + 0];
                    float inval1 = in_ptr[ic + 1];
                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];

                    if (inval0 != 0.0f && inval1 != 0.0f) {
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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) continue;
                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining4_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const float *input_f32   = (const float *)padded_input;
    float       *output_f32  = (float *)output;
    const float *bias_f32    = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    float inval0 = in_ptr[ic + 0];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;

                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f) {
                        vfloat32m8_t vwt0 = __riscv_vle32_v_f32m8(wt0, vl);
                        vfloat32m8_t vwt1 = __riscv_vle32_v_f32m8(wt1, vl);
                        vfloat32m8_t vwt2 = __riscv_vle32_v_f32m8(wt2, vl);
                        vfloat32m8_t vwt3 = __riscv_vle32_v_f32m8(wt3, vl);

                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval0, vwt0, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval1, vwt1, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval2, vwt2, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval3, vwt3, vl);
                    } else {
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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) continue;

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining8_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const float *input_f32   = (const float *)padded_input;
    float       *output_f32  = (float *)output;
    const float *bias_f32    = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    activate_store_chunk_kernel_f32m8_t activation_kernel = select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    float inval0 = in_ptr[ic + 0];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];
                    float inval4 = in_ptr[ic + 4];
                    float inval5 = in_ptr[ic + 5];
                    float inval6 = in_ptr[ic + 6];
                    float inval7 = in_ptr[ic + 7];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;

                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const float *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const float *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const float *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const float *wt7 = &weight_buffer[col_idx7 * outC + oc];

                    if (inval0 != 0.0f && inval1 != 0.0f && inval2 != 0.0f && inval3 != 0.0f &&
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
                    } else {
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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) continue;
                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

static void conv1d_fp32_vpu_chaining_nobranch_generic(NNModule *layer,
                                                      void *input,
                                                      void *output,
                                                      int unroll)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining_nobranch_generic.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m8_t activation_kernel =
        select_activate_store_chunk_kernel_f32m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m8(outC - oc);
            vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + unroll - 1 < inC; ic += unroll) {
                    for (int u = 0; u < unroll; ++u) {
                        float inval = in_ptr[ic + u];
                        int col_idx = base_col + ic + u;
                        const float *wt = &weight_buffer[col_idx * outC + oc];
                        vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                        vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                    }
                }

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_fp32_vpu_chaining2_nobranch_m8(NNModule *layer, void *input, void *output) { conv1d_fp32_vpu_chaining_nobranch_generic(layer, input, output, 2); }

void conv1d_fp32_vpu_chaining4_nobranch_m8(NNModule *layer, void *input, void *output) { conv1d_fp32_vpu_chaining_nobranch_generic(layer, input, output, 4); }

void conv1d_fp32_vpu_chaining8_nobranch_m8(NNModule *layer, void *input, void *output) { conv1d_fp32_vpu_chaining_nobranch_generic(layer, input, output, 8); }

void conv1d_fp32_vpu_chaining2_m4(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining4_m4(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining8_m4(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m4_t activation_kernel =
        select_activate_store_chunk_kernel_f32m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m4(outC - oc);
            vfloat32m4_t vacc = __riscv_vle32_v_f32m4(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];
                    float inval4 = in_ptr[ic + 4];
                    float inval5 = in_ptr[ic + 5];
                    float inval6 = in_ptr[ic + 6];
                    float inval7 = in_ptr[ic + 7];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const float *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const float *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const float *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const float *wt7 = &weight_buffer[col_idx7 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m4_t vwt = __riscv_vle32_v_f32m4(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}
void conv1d_fp32_vpu_m2(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = k * inC + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_im2col_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
            }

            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

/*
 * Register-file pressure estimate:
 *   reuse_w_2_m2: (2 + 1) * LMUL2 = 6 regs, low pressure.
 *   reuse_w_4_m2: (4 + 1) * LMUL2 = 10 regs, low pressure.
 *   reuse_w_8_m2: (8 + 1) * LMUL2 = 18 regs, within 32-reg VRF.
 */
void conv1d_fp32_vpu_im2col_reuse_w_2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 4) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc3 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval3, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 8) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        const float *row2 = &input_f32[(pos + 2) * cols];
        const float *row3 = &input_f32[(pos + 3) * cols];
        const float *row4 = &input_f32[(pos + 4) * cols];
        const float *row5 = &input_f32[(pos + 5) * cols];
        const float *row6 = &input_f32[(pos + 6) * cols];
        const float *row7 = &input_f32[(pos + 7) * cols];

        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc3 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc4 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc5 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc6 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc7 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                float inval2 = row2[col_idx];
                float inval3 = row3[col_idx];
                float inval4 = row4[col_idx];
                float inval5 = row5[col_idx];
                float inval6 = row6[col_idx];
                float inval7 = row7[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f &&
                    inval2 == 0.0f && inval3 == 0.0f &&
                    inval4 == 0.0f && inval5 == 0.0f &&
                    inval6 == 0.0f && inval7 == 0.0f) {
                    continue;
                }

                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) {
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                }
                if (inval1 != 0.0f) {
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
                if (inval2 != 0.0f) {
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval2, vwt, vl);
                }
                if (inval3 != 0.0f) {
                    vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval3, vwt, vl);
                }
                if (inval4 != 0.0f) {
                    vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval4, vwt, vl);
                }
                if (inval5 != 0.0f) {
                    vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval5, vwt, vl);
                }
                if (inval6 != 0.0f) {
                    vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval6, vwt, vl);
                }
                if (inval7 != 0.0f) {
                    vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval7, vwt, vl);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &output_f32[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &output_f32[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &output_f32[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &output_f32[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &output_f32[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &output_f32[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                for (int u = 0; u < 2; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                for (int u = 0; u < 4; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                for (int u = 0; u < 8; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt = &weight_buffer[(col_idx + u) * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                    if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                if (inval0 != 0.0f) vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0.0f) vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &output_f32[pos * outC + oc], vl);
            act_kernel(vacc1, &output_f32[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll2_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m2_t vacc00 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc01 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc10 = vacc00;
            vfloat32m2_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                for (int u = 0; u < 2; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll4_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m2_t vacc00 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc01 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc10 = vacc00;
            vfloat32m2_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                for (int u = 0; u < 4; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_fp32_vpu_im2col_reuse_w_2_unroll8_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; pos += 2) {
        const float *row0 = &input_f32[pos * cols];
        const float *row1 = &input_f32[(pos + 1) * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vfloat32m2_t vacc00 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc01 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc10 = vacc00;
            vfloat32m2_t vacc11 = vacc01;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vacc11 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            }

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                for (int u = 0; u < 8; ++u) {
                    float inval0 = row0[col_idx + u];
                    float inval1 = row1[col_idx + u];
                    if (inval0 == 0.0f && inval1 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + u) * outC + bases[0]];
                    vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                    if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                    if (blocks > 1) {
                        const float *wt1 = &weight_buffer[(col_idx + u) * outC + bases[1]];
                        vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                        if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                        if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval0 = row0[col_idx];
                float inval1 = row1[col_idx];
                if (inval0 == 0.0f && inval1 == 0.0f) continue;
                const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                vfloat32m2_t vwt0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                if (inval0 != 0.0f) vacc00 = __riscv_vfmacc_vf_f32m2(vacc00, inval0, vwt0, vls[0]);
                if (inval1 != 0.0f) vacc01 = __riscv_vfmacc_vf_f32m2(vacc01, inval1, vwt0, vls[0]);
                if (blocks > 1) {
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t vwt1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    if (inval0 != 0.0f) vacc10 = __riscv_vfmacc_vf_f32m2(vacc10, inval0, vwt1, vls[1]);
                    if (inval1 != 0.0f) vacc11 = __riscv_vfmacc_vf_f32m2(vacc11, inval1, vwt1, vls[1]);
                }
            }

            act_kernel(vacc00, &output_f32[pos * outC + bases[0]], vls[0]);
            act_kernel(vacc01, &output_f32[(pos + 1) * outC + bases[0]], vls[0]);
            if (blocks > 1) {
                act_kernel(vacc10, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc11, &output_f32[(pos + 1) * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
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
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                for (int col_idx = 0; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation); 

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            switch (blocks) {
            case 1: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v1, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
                vfloat32m2_t vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    if (inval0 == 0.0f && inval1 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt20, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt21, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval1, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt2, vls[2]);
                    vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll2_acc8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll2_acc8");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc1 = vacc0;
            vfloat32m2_t vacc2 = vacc0;
            vfloat32m2_t vacc3 = vacc0;
            vfloat32m2_t vacc4 = vacc0;
            vfloat32m2_t vacc5 = vacc0;
            vfloat32m2_t vacc6 = vacc0;
            vfloat32m2_t vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m2(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m2(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m2(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m2(&bias_f32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                if (inval0 == 0.0f && inval1 == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v, vls[0]);
                    }
                }
                if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v, vls[1]);
                    }
                }
                if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v, vls[2]);
                    }
                }
                if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval0, v, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval1, v, vls[3]);
                    }
                }
                if (blocks > 4) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval0, v, vls[4]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval1, v, vls[4]);
                    }
                }
                if (blocks > 5) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval0, v, vls[5]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval1, v, vls[5]);
                    }
                }
                if (blocks > 6) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval0, v, vls[6]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval1, v, vls[6]);
                    }
                }
                if (blocks > 7) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval0, v, vls[7]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval1, v, vls[7]);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[1]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[2]);
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[3]);
                    vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[4]);
                    vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[5]);
                    vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[6]);
                    vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[7]);
                    vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
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
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt0, vl);
                        vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, v, vl);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt1, vl);
                        vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, v, vl);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt2, vl);
                        vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, v, vl);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt3, vl);
                        vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, v, vl);
                    }
                }
            }
            for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) {
                    continue;
                }
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc2");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {0};
            size_t vls[2] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt02, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt03, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc4");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            switch (blocks) {
            case 1: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v1, vls[0]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt2, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v2, vls[0]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt3, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v3, vls[0]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                break;
            }
            case 2: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt02, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt12, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v1, vls[1]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt03, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt13, vls[1]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v1, vls[1]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                break;
            }
            case 3: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt20, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v2, vls[2]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt21, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v2, vls[2]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt02, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt12, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt22, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval2, v2, vls[2]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt03, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt13, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt23, vls[2]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval3, v2, vls[2]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt2, vls[2]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v2, vls[2]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                break;
            }
            case 4: {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                vfloat32m2_t vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
                vfloat32m2_t vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
                int col_idx = 0;
                for (; col_idx + 3 < cols; col_idx += 4) {
                    float inval0 = input_row[col_idx + 0];
                    float inval1 = input_row[col_idx + 1];
                    float inval2 = input_row[col_idx + 2];
                    float inval3 = input_row[col_idx + 3];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) {
                        continue;
                    }
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt20 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const float *wt21 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const float *wt22 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const float *wt23 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const float *wt30 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const float *wt31 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const float *wt32 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const float *wt33 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt20, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt30, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval0, v3, vls[3]);
                    }
                    if (inval1 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt21, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt31, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval1, v3, vls[3]);
                    }
                    if (inval2 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt02, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt12, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt22, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt32, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval2, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval2, v3, vls[3]);
                    }
                    if (inval3 != 0.0f) {
                        vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt03, vls[0]);
                        vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt13, vls[1]);
                        vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt23, vls[2]);
                        vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt33, vls[3]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v0, vls[0]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v1, vls[1]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval3, v2, vls[2]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval3, v3, vls[3]);
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) {
                        continue;
                    }
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    const float *wt2 = &weight_buffer[col_idx * outC + bases[2]];
                    const float *wt3 = &weight_buffer[col_idx * outC + bases[3]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vfloat32m2_t v2 = __riscv_vle32_v_f32m2(wt2, vls[2]);
                    vfloat32m2_t v3 = __riscv_vle32_v_f32m2(wt3, vls[3]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                    vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v2, vls[2]);
                    vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval, v3, vls[3]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
                act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
                act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
                break;
            }
            default:
                break;
            }
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll4_acc8_m2(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll4_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0, vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m2(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m2(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m2(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m2(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval0,v,vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval1,v,vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval2,v,vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[0]);
                            vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval3,v,vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval0,v,vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval1,v,vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval2,v,vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[1]);
                            vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval3,v,vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval0,v,vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval1,v,vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval2,v,vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[2]);
                            vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval3,v,vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval0,v,vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval1,v,vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval2,v,vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[3]);
                            vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval3,v,vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m2(vacc4,inval0,v,vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m2(vacc4,inval1,v,vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m2(vacc4,inval2,v,vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[4]);
                            vacc4=__riscv_vfmacc_vf_f32m2(vacc4,inval3,v,vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m2(vacc5,inval0,v,vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m2(vacc5,inval1,v,vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m2(vacc5,inval2,v,vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[5]);
                            vacc5=__riscv_vfmacc_vf_f32m2(vacc5,inval3,v,vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m2(vacc6,inval0,v,vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m2(vacc6,inval1,v,vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m2(vacc6,inval2,v,vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[6]);
                            vacc6=__riscv_vfmacc_vf_f32m2(vacc6,inval3,v,vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m2(vacc7,inval0,v,vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m2(vacc7,inval1,v,vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m2(vacc7,inval2,v,vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[7]);
                            vacc7=__riscv_vfmacc_vf_f32m2(vacc7,inval3,v,vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval,v,vls[0]);
                    } else if (b == 1) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval,v,vls[1]);
                    } else if (b == 2) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval,v,vls[2]);
                    } else if (b == 3) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval,v,vls[3]);
                    } else if (b == 4) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[4]);
                        vacc4=__riscv_vfmacc_vf_f32m2(vacc4,inval,v,vls[4]);
                    } else if (b == 5) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[5]);
                        vacc5=__riscv_vfmacc_vf_f32m2(vacc5,inval,v,vls[5]);
                    } else if (b == 6) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[6]);
                        vacc6=__riscv_vfmacc_vf_f32m2(vacc6,inval,v,vls[6]);
                    } else {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[7]);
                        vacc7=__riscv_vfmacc_vf_f32m2(vacc7,inval,v,vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_m2(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0];
                float inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2];
                float inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4];
                float inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6];
                float inval7 = input_row[col_idx + 7];
                const float *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const float *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const float *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const float *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const float *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const float *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const float *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const float *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];
                if (inval0 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt0, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval0, v, vl);
                } if (inval1 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt1, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval1, v, vl);
                } if (inval2 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt2, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval2, v, vl);
                } if (inval3 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt3, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval3, v, vl);
                } if (inval4 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt4, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval4, v, vl);
                } if (inval5 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt5, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval5, v, vl);
                } if (inval6 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt6, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval6, v, vl);
                } if (inval7 != 0.0f) {
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt7, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval7, v, vl);
                } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                const float *wt = &weight_buffer[col_idx * outC + oc];
                vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
            } act_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc2_m2(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc2");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[2] = {
                0};
            size_t vls[2] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 2) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }
            if (blocks == 1) {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt0, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt1, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt2, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt3, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt4, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt5, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt6, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0.0f) { vfloat32m2_t v = __riscv_vle32_v_f32m2(wt7, vls[0]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval7, v, vls[0]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            } else {
                vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
                vfloat32m2_t vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
                int col_idx = 0;
                for (; col_idx + 7 < cols; col_idx += 8) {
                    float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1], inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                    float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5], inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                    if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f &&
                        inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                    const float *wt00 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const float *wt01 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const float *wt02 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const float *wt03 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const float *wt04 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const float *wt05 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const float *wt06 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const float *wt07 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    const float *wt10 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const float *wt11 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const float *wt12 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const float *wt13 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const float *wt14 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const float *wt15 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const float *wt16 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const float *wt17 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt00, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt10, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v1, vls[1]); }
                    if (inval1 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt01, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt11, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v1, vls[1]); }
                    if (inval2 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt02, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt12, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v1, vls[1]); }
                    if (inval3 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt03, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt13, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v1, vls[1]); }
                    if (inval4 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt04, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt14, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval4, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval4, v1, vls[1]); }
                    if (inval5 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt05, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt15, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval5, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval5, v1, vls[1]); }
                    if (inval6 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt06, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt16, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval6, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval6, v1, vls[1]); }
                    if (inval7 != 0.0f) { vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt07, vls[0]); vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt17, vls[1]); vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval7, v0, vls[0]); vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval7, v1, vls[1]); }
                }
                for (; col_idx < cols; ++col_idx) {
                    float inval = input_row[col_idx];
                    if (inval == 0.0f) continue;
                    const float *wt0 = &weight_buffer[col_idx * outC + bases[0]];
                    const float *wt1 = &weight_buffer[col_idx * outC + bases[1]];
                    vfloat32m2_t v0 = __riscv_vle32_v_f32m2(wt0, vls[0]);
                    vfloat32m2_t v1 = __riscv_vle32_v_f32m2(wt1, vls[1]);
                    vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v0, vls[0]);
                    vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v1, vls[1]);
                }
                act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
                act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            }
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc4_m2(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc4");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {
                0};
            size_t vls[4] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx+0], inval1 = input_row[col_idx+1], inval2 = input_row[col_idx+2], inval3 = input_row[col_idx+3], inval4 = input_row[col_idx+4], inval5 = input_row[col_idx+5], inval6 = input_row[col_idx+6], inval7 = input_row[col_idx+7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                if (blocks > 0) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[0]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[0]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[0]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[0]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[0]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[0]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[0]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[0]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval0,v,vls[0]);
                    } if (inval1 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval1,v,vls[0]);
                    } if (inval2 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval2,v,vls[0]);
                    } if (inval3 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval3,v,vls[0]);
                    } if (inval4 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w4,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval4,v,vls[0]);
                    } if (inval5 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w5,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval5,v,vls[0]);
                    } if (inval6 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w6,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval6,v,vls[0]);
                    } if (inval7 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w7,vls[0]);
                        vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval7,v,vls[0]);
                    } } if (blocks > 1) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[1]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[1]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[1]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[1]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[1]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[1]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[1]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[1]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval0,v,vls[1]);
                    } if (inval1 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval1,v,vls[1]);
                    } if (inval2 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval2,v,vls[1]);
                    } if (inval3 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval3,v,vls[1]);
                    } if (inval4 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w4,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval4,v,vls[1]);
                    } if (inval5 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w5,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval5,v,vls[1]);
                    } if (inval6 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w6,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval6,v,vls[1]);
                    } if (inval7 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w7,vls[1]);
                        vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval7,v,vls[1]);
                    } } if (blocks > 2) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[2]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[2]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[2]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[2]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[2]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[2]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[2]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[2]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval0,v,vls[2]);
                    } if (inval1 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval1,v,vls[2]);
                    } if (inval2 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval2,v,vls[2]);
                    } if (inval3 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval3,v,vls[2]);
                    } if (inval4 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w4,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval4,v,vls[2]);
                    } if (inval5 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w5,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval5,v,vls[2]);
                    } if (inval6 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w6,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval6,v,vls[2]);
                    } if (inval7 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w7,vls[2]);
                        vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval7,v,vls[2]);
                    } } if (blocks > 3) {
                    const float *w0 = &weight_buffer[(col_idx+0)*outC+bases[3]];
                    const float *w1 = &weight_buffer[(col_idx+1)*outC+bases[3]];
                    const float *w2 = &weight_buffer[(col_idx+2)*outC+bases[3]];
                    const float *w3 = &weight_buffer[(col_idx+3)*outC+bases[3]];
                    const float *w4 = &weight_buffer[(col_idx+4)*outC+bases[3]];
                    const float *w5 = &weight_buffer[(col_idx+5)*outC+bases[3]];
                    const float *w6 = &weight_buffer[(col_idx+6)*outC+bases[3]];
                    const float *w7 = &weight_buffer[(col_idx+7)*outC+bases[3]];
                    if (inval0 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w0,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval0,v,vls[3]);
                    } if (inval1 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w1,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval1,v,vls[3]);
                    } if (inval2 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w2,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval2,v,vls[3]);
                    } if (inval3 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w3,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval3,v,vls[3]);
                    } if (inval4 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w4,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval4,v,vls[3]);
                    } if (inval5 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w5,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval5,v,vls[3]);
                    } if (inval6 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w6,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval6,v,vls[3]);
                    } if (inval7 != 0.0f) {
                        vfloat32m2_t v=__riscv_vle32_v_f32m2(w7,vls[3]);
                        vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval7,v,vls[3]);
                    } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                if (blocks > 0) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[0]];
                    vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[0]);
                    vacc0=__riscv_vfmacc_vf_f32m2(vacc0,inval,v,vls[0]);
                } if (blocks > 1) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[1]];
                    vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[1]);
                    vacc1=__riscv_vfmacc_vf_f32m2(vacc1,inval,v,vls[1]);
                } if (blocks > 2) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[2]];
                    vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[2]);
                    vacc2=__riscv_vfmacc_vf_f32m2(vacc2,inval,v,vls[2]);
                } if (blocks > 3) {
                    const float *wt = &weight_buffer[col_idx*outC+bases[3]];
                    vfloat32m2_t v=__riscv_vle32_v_f32m2(wt,vls[3]);
                    vacc3=__riscv_vfmacc_vf_f32m2(vacc3,inval,v,vls[3]);
                } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_im2col_unroll8_acc8_m2(NNModule *layer, void *input, void *output) {
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\\n", "conv1d_fp32_vpu_im2col_unroll8_acc8");
        exit(EXIT_FAILURE);
    } int outW = layer->outputShape.W, outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const float *input_f32 = (const float *)im2col_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t act_kernel = select_activate_store_chunk_kernel_f32m2(layer->activation);
    for (int pos = 0; pos < outW; ++pos) {
        const float *input_row = &input_f32[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {
                0};
            size_t vls[8] = {
                0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e32m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            } vfloat32m2_t vacc0 = __riscv_vle32_v_f32m2(&bias_f32[bases[0]], vls[0]);
            vfloat32m2_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            vfloat32m2_t vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            if (blocks > 1) vacc1 = __riscv_vle32_v_f32m2(&bias_f32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_f32m2(&bias_f32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_f32m2(&bias_f32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_f32m2(&bias_f32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_f32m2(&bias_f32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_f32m2(&bias_f32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_f32m2(&bias_f32[bases[7]], vls[7]);
            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                float inval0 = input_row[col_idx + 0], inval1 = input_row[col_idx + 1];
                float inval2 = input_row[col_idx + 2], inval3 = input_row[col_idx + 3];
                float inval4 = input_row[col_idx + 4], inval5 = input_row[col_idx + 5];
                float inval6 = input_row[col_idx + 6], inval7 = input_row[col_idx + 7];
                if (inval0 == 0.0f && inval1 == 0.0f && inval2 == 0.0f && inval3 == 0.0f && inval4 == 0.0f && inval5 == 0.0f && inval6 == 0.0f && inval7 == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *w0 = &weight_buffer[(col_idx + 0) * outC + bases[b]];
                    const float *w1 = &weight_buffer[(col_idx + 1) * outC + bases[b]];
                    const float *w2 = &weight_buffer[(col_idx + 2) * outC + bases[b]];
                    const float *w3 = &weight_buffer[(col_idx + 3) * outC + bases[b]];
                    const float *w4 = &weight_buffer[(col_idx + 4) * outC + bases[b]];
                    const float *w5 = &weight_buffer[(col_idx + 5) * outC + bases[b]];
                    const float *w6 = &weight_buffer[(col_idx + 6) * outC + bases[b]];
                    const float *w7 = &weight_buffer[(col_idx + 7) * outC + bases[b]];
                    if (b == 0) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval0, v, vls[0]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval1, v, vls[0]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval2, v, vls[0]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval3, v, vls[0]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval4, v, vls[0]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval5, v, vls[0]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval6, v, vls[0]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[0]);
                            vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval7, v, vls[0]);
                        } } else if (b == 1) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval0, v, vls[1]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval1, v, vls[1]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval2, v, vls[1]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval3, v, vls[1]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval4, v, vls[1]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval5, v, vls[1]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval6, v, vls[1]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[1]);
                            vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval7, v, vls[1]);
                        } } else if (b == 2) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval0, v, vls[2]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval1, v, vls[2]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval2, v, vls[2]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval3, v, vls[2]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval4, v, vls[2]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval5, v, vls[2]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval6, v, vls[2]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[2]);
                            vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval7, v, vls[2]);
                        } } else if (b == 3) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval0, v, vls[3]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval1, v, vls[3]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval2, v, vls[3]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval3, v, vls[3]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval4, v, vls[3]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval5, v, vls[3]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval6, v, vls[3]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[3]);
                            vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval7, v, vls[3]);
                        } } else if (b == 4) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval0, v, vls[4]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval1, v, vls[4]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval2, v, vls[4]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval3, v, vls[4]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval4, v, vls[4]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval5, v, vls[4]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval6, v, vls[4]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[4]);
                            vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval7, v, vls[4]);
                        } } else if (b == 5) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval0, v, vls[5]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval1, v, vls[5]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval2, v, vls[5]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval3, v, vls[5]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval4, v, vls[5]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval5, v, vls[5]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval6, v, vls[5]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[5]);
                            vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval7, v, vls[5]);
                        } } else if (b == 6) {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval0, v, vls[6]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval1, v, vls[6]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval2, v, vls[6]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval3, v, vls[6]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval4, v, vls[6]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval5, v, vls[6]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval6, v, vls[6]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[6]);
                            vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval7, v, vls[6]);
                        } } else {
                        if (inval0 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w0, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval0, v, vls[7]);
                        } if (inval1 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w1, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval1, v, vls[7]);
                        } if (inval2 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w2, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval2, v, vls[7]);
                        } if (inval3 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w3, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval3, v, vls[7]);
                        } if (inval4 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w4, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval4, v, vls[7]);
                        } if (inval5 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w5, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval5, v, vls[7]);
                        } if (inval6 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w6, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval6, v, vls[7]);
                        } if (inval7 != 0.0f) {
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(w7, vls[7]);
                            vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval7, v, vls[7]);
                        } } } } for (; col_idx < cols; ++col_idx) {
                float inval = input_row[col_idx];
                if (inval == 0.0f) continue;
                for (int b = 0; b < blocks; ++b) {
                    const float *wt = &weight_buffer[col_idx * outC + bases[b]];
                    if (b == 0) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[0]);
                        vacc0 = __riscv_vfmacc_vf_f32m2(vacc0, inval, v, vls[0]);
                    } else if (b == 1) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[1]);
                        vacc1 = __riscv_vfmacc_vf_f32m2(vacc1, inval, v, vls[1]);
                    } else if (b == 2) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[2]);
                        vacc2 = __riscv_vfmacc_vf_f32m2(vacc2, inval, v, vls[2]);
                    } else if (b == 3) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[3]);
                        vacc3 = __riscv_vfmacc_vf_f32m2(vacc3, inval, v, vls[3]);
                    } else if (b == 4) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[4]);
                        vacc4 = __riscv_vfmacc_vf_f32m2(vacc4, inval, v, vls[4]);
                    } else if (b == 5) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[5]);
                        vacc5 = __riscv_vfmacc_vf_f32m2(vacc5, inval, v, vls[5]);
                    } else if (b == 6) {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[6]);
                        vacc6 = __riscv_vfmacc_vf_f32m2(vacc6, inval, v, vls[6]);
                    } else {
                        vfloat32m2_t v = __riscv_vle32_v_f32m2(wt, vls[7]);
                        vacc7 = __riscv_vfmacc_vf_f32m2(vacc7, inval, v, vls[7]);
                    } } } act_kernel(vacc0, &output_f32[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &output_f32[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &output_f32[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &output_f32[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &output_f32[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &output_f32[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &output_f32[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &output_f32[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        } } safe_free(im2col_input);
}

void conv1d_fp32_vpu_chaining2_m2(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining4_m2(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_fp32_vpu_chaining8_m2(NNModule *layer, void *input, void *output)
{
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_fp32_vpu_chaining8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride = layer->params.conv.stride;

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv.bias;
    const float *weight_buffer = (const float *)layer->params.conv.weights_rvv;
    activate_store_chunk_kernel_f32m2_t activation_kernel =
        select_activate_store_chunk_kernel_f32m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e32m2(outC - oc);
            vfloat32m2_t vacc = __riscv_vle32_v_f32m2(&bias_f32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const float *in_ptr = &input_f32[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    float inval0 = in_ptr[ic];
                    float inval1 = in_ptr[ic + 1];
                    float inval2 = in_ptr[ic + 2];
                    float inval3 = in_ptr[ic + 3];
                    float inval4 = in_ptr[ic + 4];
                    float inval5 = in_ptr[ic + 5];
                    float inval6 = in_ptr[ic + 6];
                    float inval7 = in_ptr[ic + 7];
                    int col_idx0 = base_col + ic;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;
                    const float *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const float *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const float *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const float *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const float *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const float *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const float *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const float *wt7 = &weight_buffer[col_idx7 * outC + oc];

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
                        if (inval4 != 0.0f) {
                            vfloat32m2_t vwt4 = __riscv_vle32_v_f32m2(wt4, vl);
                            vacc = __riscv_vfmacc_vf_f32m2(vacc, inval4, vwt4, vl);
                        }
                        if (inval5 != 0.0f) {
                            vfloat32m2_t vwt5 = __riscv_vle32_v_f32m2(wt5, vl);
                            vacc = __riscv_vfmacc_vf_f32m2(vacc, inval5, vwt5, vl);
                        }
                        if (inval6 != 0.0f) {
                            vfloat32m2_t vwt6 = __riscv_vle32_v_f32m2(wt6, vl);
                            vacc = __riscv_vfmacc_vf_f32m2(vacc, inval6, vwt6, vl);
                        }
                        if (inval7 != 0.0f) {
                            vfloat32m2_t vwt7 = __riscv_vle32_v_f32m2(wt7, vl);
                            vacc = __riscv_vfmacc_vf_f32m2(vacc, inval7, vwt7, vl);
                        }
                    }
                }
                for (; ic < inC; ++ic) {
                    float inval = in_ptr[ic];
                    if (inval == 0.0f) {
                        continue;
                    }

                    int col_idx = base_col + ic;
                    const float *wt = &weight_buffer[col_idx * outC + oc];
                    vfloat32m2_t vwt = __riscv_vle32_v_f32m2(wt, vl);
                    vacc = __riscv_vfmacc_vf_f32m2(vacc, inval, vwt, vl);
                }
            }

            activation_kernel(vacc, &output_f32[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}
