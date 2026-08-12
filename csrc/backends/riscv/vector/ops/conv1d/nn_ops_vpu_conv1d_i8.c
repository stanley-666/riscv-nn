/* SPDX-FileContributor: Person: Stanley Lee */
#include "nn_layer.h"
#include "nn_utils.h"
#include "backends/riscv/vector/ops/activation/nn_activation_int_internal.h"
#include "backends/riscv/vector/ops/activation/nn_activation_fp_internal.h"

#include "nn_ops_vpu_conv1d_i8_internal.h"
#if defined(BAREMETAL)
#include "baremetal_timer.h"
#endif

/*
 * GCC cannot infer that the RVV accumulators guarded by groups/blocks are
 * initialized on every matching guarded use.  Keep this workaround local to
 * these kernels; the guards are also what make tail output-channel groups safe.
 */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

void conv1d_i8_vpu_m2(NNModule *layer, void *input, void *output)
{
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0)
                        continue;              
                    int col_idx = k * inC + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
                }
            }
            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_i8_vpu_im2col_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            vint32m2_t vacc1 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                if (inval0 == 0 && inval1 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                if (inval0 != 0)
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0)
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    for (; pos < outW; ++pos) {
        const int8_t *row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    int pos = 0;
    for (; pos + 3 < outW; pos += 4) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        const int8_t *row2 = &input_i8[(pos + 2) * cols];
        const int8_t *row3 = &input_i8[(pos + 3) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            vint32m2_t vacc1 = vacc0;
            vint32m2_t vacc2 = vacc0;
            vint32m2_t vacc3 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                int16_t inval2 = (int16_t)row2[col_idx];
                int16_t inval3 = (int16_t)row3[col_idx];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                if (inval0 != 0)
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0)
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt, vl);
                if (inval2 != 0)
                    vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, vwt, vl);
                if (inval3 != 0)
                    vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW) {
        conv1d_i8_vpu_im2col_m2(layer, input, output);
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    int pos = 0;
    for (; pos + 7 < outW; pos += 8) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        const int8_t *row2 = &input_i8[(pos + 2) * cols];
        const int8_t *row3 = &input_i8[(pos + 3) * cols];
        const int8_t *row4 = &input_i8[(pos + 4) * cols];
        const int8_t *row5 = &input_i8[(pos + 5) * cols];
        const int8_t *row6 = &input_i8[(pos + 6) * cols];
        const int8_t *row7 = &input_i8[(pos + 7) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            vint32m2_t vacc1 = vacc0;
            vint32m2_t vacc2 = vacc0;
            vint32m2_t vacc3 = vacc0;
            vint32m2_t vacc4 = vacc0;
            vint32m2_t vacc5 = vacc0;
            vint32m2_t vacc6 = vacc0;
            vint32m2_t vacc7 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                int16_t inval2 = (int16_t)row2[col_idx];
                int16_t inval3 = (int16_t)row3[col_idx];
                int16_t inval4 = (int16_t)row4[col_idx];
                int16_t inval5 = (int16_t)row5[col_idx];
                int16_t inval6 = (int16_t)row6[col_idx];
                int16_t inval7 = (int16_t)row7[col_idx];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                if (inval0 != 0) vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt, vl);
                if (inval1 != 0) vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt, vl);
                if (inval2 != 0) vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, vwt, vl);
                if (inval3 != 0) vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, vwt, vl);
                if (inval4 != 0) vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval4, vwt, vl);
                if (inval5 != 0) vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval5, vwt, vl);
                if (inval6 != 0) vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval6, vwt, vl);
                if (inval7 != 0) vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval7, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &M[oc], &Z[oc], &output_i8[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &M[oc], &Z[oc], &output_i8[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &M[oc], &Z[oc], &output_i8[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &M[oc], &Z[oc], &output_i8[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW) {
        conv1d_i8_vpu_im2col_m2(layer, input, output);
    }

    safe_free(im2col_input);
}

static void conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(NNModule *layer, void *input, void *output,
                                                           int unroll_cols, int acc2,
                                                           const char *kernel_name)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\n", kernel_name);
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];

        if (!acc2) {
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e16m1(outC - oc);
                vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
                vint32m2_t vacc1 = vacc0;

                for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                    int block = cols - col_idx;
                    if (block > unroll_cols)
                        block = unroll_cols;
                    for (int u = 0; u < block; ++u) {
                        int k = col_idx + u;
                        int16_t inval0 = (int16_t)row0[k];
                        int16_t inval1 = (int16_t)row1[k];
                        if (inval0 == 0 && inval1 == 0)
                            continue;
                        const int16_t *wt = &weight_buffer[k * outC + oc];
                        vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                        if (inval0 != 0)
                            vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt, vl);
                        if (inval1 != 0)
                            vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt, vl);
                    }
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
                act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
                oc += (int)vl;
            }
            continue;
        }

        int oc = 0;
        while (oc < outC) {
            int base0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m1(outC - base0);
            int base1 = base0 + (int)vl0;
            int blocks = 1;
            size_t vl1 = 0;
            if (base1 < outC) {
                vl1 = __riscv_vsetvl_e16m1(outC - base1);
                blocks = 2;
            }

            vint32m2_t vacc00 = __riscv_vle32_v_i32m2(&bias_i32[base0], vl0);
            vint32m2_t vacc01 = vacc00;
            vint32m2_t vacc10 = vacc00;
            vint32m2_t vacc11 = vacc00;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_i32m2(&bias_i32[base1], vl1);
                vacc11 = vacc10;
            }

            for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                int block = cols - col_idx;
                if (block > unroll_cols)
                    block = unroll_cols;
                for (int u = 0; u < block; ++u) {
                    int k = col_idx + u;
                    int16_t inval0 = (int16_t)row0[k];
                    int16_t inval1 = (int16_t)row1[k];
                    if (inval0 == 0 && inval1 == 0)
                        continue;
                    const int16_t *wt0 = &weight_buffer[k * outC + base0];
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                    if (inval0 != 0)
                        vacc00 = __riscv_vwmacc_vx_i32m2(vacc00, inval0, vwt0, vl0);
                    if (inval1 != 0)
                        vacc01 = __riscv_vwmacc_vx_i32m2(vacc01, inval1, vwt0, vl0);
                    if (blocks > 1) {
                        const int16_t *wt1 = &weight_buffer[k * outC + base1];
                        vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl1);
                        if (inval0 != 0)
                            vacc10 = __riscv_vwmacc_vx_i32m2(vacc10, inval0, vwt1, vl1);
                        if (inval1 != 0)
                            vacc11 = __riscv_vwmacc_vx_i32m2(vacc11, inval1, vwt1, vl1);
                    }
                }
            }

            act_kernel(vacc00, &M[base0], &Z[base0], &output_i8[pos * outC + base0], vl0);
            act_kernel(vacc01, &M[base0], &Z[base0], &output_i8[(pos + 1) * outC + base0], vl0);
            if (blocks > 1) {
                act_kernel(vacc10, &M[base1], &Z[base1], &output_i8[pos * outC + base1], vl1);
                act_kernel(vacc11, &M[base1], &Z[base1], &output_i8[(pos + 1) * outC + base1], vl1);
            }
            oc = base1 + (int)vl1;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m2(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 2, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m2");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 2, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m2");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 4, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m2");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 4, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m2");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 8, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m2");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m2(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m2_impl(layer, input, output, 8, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m2");
}

void conv1d_i8_vpu_im2col_unroll2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    if (inval0 != 0 && inval1 != 0) {
                        vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                        vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt1, vl0);
                    } else {
                        if (inval0 != 0) { vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt1, vl0); }
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt, vl0);
                }
                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m1(rem1);
            int oc1 = oc + (int)vl0;
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
            vint32m2_t vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;

                if (inval0 != 0 && inval1 != 0) {
                    vint16m1_t vwt00 = __riscv_vle16_v_i16m1(wt00, vl0);
                    vint16m1_t vwt10 = __riscv_vle16_v_i16m1(wt10, vl0);
                    vint16m1_t vwt01 = __riscv_vle16_v_i16m1(wt01, vl1);
                    vint16m1_t vwt11 = __riscv_vle16_v_i16m1(wt11, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt10, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt11, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m1_t vwt00 = __riscv_vle16_v_i16m1(wt00, vl0);
                        vint16m1_t vwt01 = __riscv_vle16_v_i16m1(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m1_t vwt10 = __riscv_vle16_v_i16m1(wt10, vl0);
                        vint16m1_t vwt11 = __riscv_vle16_v_i16m1(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt11, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc);
            int oc0 = oc;
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0;
            vint32m2_t vacc1, vacc2, vacc3;

            if (oc_cursor < outC) {
                groups = 2;
                oc1 = oc_cursor;
                vl1 = __riscv_vsetvl_e16m1(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1);
                oc_cursor = oc1 + (int)vl1;
            }
            if (oc_cursor < outC) {
                groups = 3;
                oc2 = oc_cursor;
                vl2 = __riscv_vsetvl_e16m1(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m2(&bias_i32[oc2], vl2);
                oc_cursor = oc2 + (int)vl2;
            }
            if (oc_cursor < outC) {
                groups = 4;
                oc3 = oc_cursor;
                vl3 = __riscv_vsetvl_e16m1(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m2(&bias_i32[oc3], vl3);
                oc_cursor = oc3 + (int)vl3;
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base0 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, vwt3, vl3); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base1 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, vwt3, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base + oc0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, vwt3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc);
            int oc0 = oc;
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0, vl4 = 0, vl5 = 0, vl6 = 0, vl7 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0, oc4 = 0, oc5 = 0, oc6 = 0, oc7 = 0;
            vint32m2_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;

            if (oc_cursor < outC) { groups = 2; oc1 = oc_cursor; vl1 = __riscv_vsetvl_e16m1(outC - oc1); vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1); oc_cursor = oc1 + (int)vl1; }
            if (oc_cursor < outC) { groups = 3; oc2 = oc_cursor; vl2 = __riscv_vsetvl_e16m1(outC - oc2); vacc2 = __riscv_vle32_v_i32m2(&bias_i32[oc2], vl2); oc_cursor = oc2 + (int)vl2; }
            if (oc_cursor < outC) { groups = 4; oc3 = oc_cursor; vl3 = __riscv_vsetvl_e16m1(outC - oc3); vacc3 = __riscv_vle32_v_i32m2(&bias_i32[oc3], vl3); oc_cursor = oc3 + (int)vl3; }
            if (oc_cursor < outC) { groups = 5; oc4 = oc_cursor; vl4 = __riscv_vsetvl_e16m1(outC - oc4); vacc4 = __riscv_vle32_v_i32m2(&bias_i32[oc4], vl4); oc_cursor = oc4 + (int)vl4; }
            if (oc_cursor < outC) { groups = 6; oc5 = oc_cursor; vl5 = __riscv_vsetvl_e16m1(outC - oc5); vacc5 = __riscv_vle32_v_i32m2(&bias_i32[oc5], vl5); oc_cursor = oc5 + (int)vl5; }
            if (oc_cursor < outC) { groups = 7; oc6 = oc_cursor; vl6 = __riscv_vsetvl_e16m1(outC - oc6); vacc6 = __riscv_vle32_v_i32m2(&bias_i32[oc6], vl6); oc_cursor = oc6 + (int)vl6; }
            if (oc_cursor < outC) { groups = 8; oc7 = oc_cursor; vl7 = __riscv_vsetvl_e16m1(outC - oc7); vacc7 = __riscv_vle32_v_i32m2(&bias_i32[oc7], vl7); oc_cursor = oc7 + (int)vl7; }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base0 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, vwt3, vl3); }
                    if (groups > 4) { vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt_base0 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval0, vwt4, vl4); }
                    if (groups > 5) { vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt_base0 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval0, vwt5, vl5); }
                    if (groups > 6) { vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt_base0 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval0, vwt6, vl6); }
                    if (groups > 7) { vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt_base0 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval0, vwt7, vl7); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base1 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, vwt3, vl3); }
                    if (groups > 4) { vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt_base1 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval1, vwt4, vl4); }
                    if (groups > 5) { vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt_base1 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval1, vwt5, vl5); }
                    if (groups > 6) { vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt_base1 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval1, vwt6, vl6); }
                    if (groups > 7) { vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt_base1 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval1, vwt7, vl7); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt_base + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, vwt3, vl3); }
                if (groups > 4) { vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt_base + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval, vwt4, vl4); }
                if (groups > 5) { vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt_base + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval, vwt5, vl5); }
                if (groups > 6) { vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt_base + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval, vwt6, vl6); }
                if (groups > 7) { vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt_base + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval, vwt7, vl7); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            if (groups > 4) act_kernel(vacc4, &M[oc4], &Z[oc4], &output_i8[pos * outC + oc4], vl4);
            if (groups > 5) act_kernel(vacc5, &M[oc5], &Z[oc5], &output_i8[pos * outC + oc5], vl5);
            if (groups > 6) act_kernel(vacc6, &M[oc6], &Z[oc6], &output_i8[pos * outC + oc6], vl6);
            if (groups > 7) act_kernel(vacc7, &M[oc7], &Z[oc7], &output_i8[pos * outC + oc7], vl7);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 3 < cols; col_idx += 4) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];

                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

                    if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                        vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                        vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl0);
                        vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl0);
                        vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, vwt2, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, vwt3, vl0);
                    } else {
                        if (inval0 != 0) { vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt1, vl0); }
                        if (inval2 != 0) { vint16m1_t vwt2 = __riscv_vle16_v_i16m1(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, vwt2, vl0); }
                        if (inval3 != 0) { vint16m1_t vwt3 = __riscv_vle16_v_i16m1(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, vwt3, vl0); }
                    }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m1(rem1);
            int oc1 = oc + (int)vl0;
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
            vint32m2_t vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;

                if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                    vint16m1_t vwt00 = __riscv_vle16_v_i16m1(wt00, vl0);
                    vint16m1_t vwt10 = __riscv_vle16_v_i16m1(wt10, vl0);
                    vint16m1_t vwt20 = __riscv_vle16_v_i16m1(wt20, vl0);
                    vint16m1_t vwt30 = __riscv_vle16_v_i16m1(wt30, vl0);
                    vint16m1_t vwt01 = __riscv_vle16_v_i16m1(wt01, vl1);
                    vint16m1_t vwt11 = __riscv_vle16_v_i16m1(wt11, vl1);
                    vint16m1_t vwt21 = __riscv_vle16_v_i16m1(wt21, vl1);
                    vint16m1_t vwt31 = __riscv_vle16_v_i16m1(wt31, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt10, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, vwt20, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, vwt30, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt11, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, vwt21, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, vwt31, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m1_t vwt00 = __riscv_vle16_v_i16m1(wt00, vl0);
                        vint16m1_t vwt01 = __riscv_vle16_v_i16m1(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m1_t vwt10 = __riscv_vle16_v_i16m1(wt10, vl0);
                        vint16m1_t vwt11 = __riscv_vle16_v_i16m1(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, vwt11, vl1);
                    }
                    if (inval2 != 0) {
                        vint16m1_t vwt20 = __riscv_vle16_v_i16m1(wt20, vl0);
                        vint16m1_t vwt21 = __riscv_vle16_v_i16m1(wt21, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, vwt20, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, vwt21, vl1);
                    }
                    if (inval3 != 0) {
                        vint16m1_t vwt30 = __riscv_vle16_v_i16m1(wt30, vl0);
                        vint16m1_t vwt31 = __riscv_vle16_v_i16m1(wt31, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, vwt30, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, vwt31, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int oc0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc0);
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc0], vl0);

            int oc1 = oc0 + (int)vl0;
            size_t vl1 = 0;
            vint32m2_t vacc1;
            int have1 = oc1 < outC;
            if (have1) {
                vl1 = __riscv_vsetvl_e16m1(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1);
            }

            int oc2 = oc1 + (int)vl1;
            size_t vl2 = 0;
            vint32m2_t vacc2;
            int have2 = oc2 < outC;
            if (have2) {
                vl2 = __riscv_vsetvl_e16m1(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m2(&bias_i32[oc2], vl2);
            }

            int oc3 = oc2 + (int)vl2;
            size_t vl3 = 0;
            vint32m2_t vacc3;
            int have3 = oc3 < outC;
            if (have3) {
                vl3 = __riscv_vsetvl_e16m1(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m2(&bias_i32[oc3], vl3);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc0];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc0];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc0];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc0];
                if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt00, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v, vl0); }
                if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt10, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v, vl0); }
                if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt20, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v, vl0); }
                if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt30, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v, vl0); }

                if (have1) {
                    const int16_t *wt01 = &weight_buffer[(col_idx + 0) * outC + oc1];
                    const int16_t *wt11 = &weight_buffer[(col_idx + 1) * outC + oc1];
                    const int16_t *wt21 = &weight_buffer[(col_idx + 2) * outC + oc1];
                    const int16_t *wt31 = &weight_buffer[(col_idx + 3) * outC + oc1];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt01, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, v, vl1); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt11, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, v, vl1); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt21, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, v, vl1); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt31, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, v, vl1); }
                }
                if (have2) {
                    const int16_t *wt02 = &weight_buffer[(col_idx + 0) * outC + oc2];
                    const int16_t *wt12 = &weight_buffer[(col_idx + 1) * outC + oc2];
                    const int16_t *wt22 = &weight_buffer[(col_idx + 2) * outC + oc2];
                    const int16_t *wt32 = &weight_buffer[(col_idx + 3) * outC + oc2];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt02, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, v, vl2); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt12, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, v, vl2); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt22, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, v, vl2); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt32, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval3, v, vl2); }
                }
                if (have3) {
                    const int16_t *wt03 = &weight_buffer[(col_idx + 0) * outC + oc3];
                    const int16_t *wt13 = &weight_buffer[(col_idx + 1) * outC + oc3];
                    const int16_t *wt23 = &weight_buffer[(col_idx + 2) * outC + oc3];
                    const int16_t *wt33 = &weight_buffer[(col_idx + 3) * outC + oc3];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt03, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, v, vl3); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt13, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, v, vl3); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt23, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval2, v, vl3); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt33, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, v, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc0];
                vint16m1_t v0 = __riscv_vle16_v_i16m1(wt0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, v0, vl0);
                if (have1) { const int16_t *wt1 = &weight_buffer[col_idx * outC + oc1]; vint16m1_t v1 = __riscv_vle16_v_i16m1(wt1, vl1); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, v1, vl1); }
                if (have2) { const int16_t *wt2 = &weight_buffer[col_idx * outC + oc2]; vint16m1_t v2 = __riscv_vle16_v_i16m1(wt2, vl2); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, v2, vl2); }
                if (have3) { const int16_t *wt3 = &weight_buffer[col_idx * outC + oc3]; vint16m1_t v3 = __riscv_vle16_v_i16m1(wt3, vl3); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, v3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (have1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (have2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (have3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc3 + (int)vl3;
            if (!have3)
                oc = have2 ? (oc2 + (int)vl2) : (have1 ? (oc1 + (int)vl1) : (oc0 + (int)vl0));
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8];
            size_t vls[8];
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m1(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[bases[0]], vls[0]);
            vint32m2_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m2(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m2(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m2(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m2(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m2(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m2(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m2(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;

                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval3, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval3, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval3, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval3, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval3, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[0]);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[1]);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[2]);
                    vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[3]);
                    vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[4]);
                    vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[5]);
                    vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[6]);
                    vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[7]);
                    vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m1_t vwt16 = __riscv_vle16_v_i16m1(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc2_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc2_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m1(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 7 < cols; col_idx += 8) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];
                    int16_t inval4 = (int16_t)input_row[col_idx + 4];
                    int16_t inval5 = (int16_t)input_row[col_idx + 5];
                    int16_t inval6 = (int16_t)input_row[col_idx + 6];
                    int16_t inval7 = (int16_t)input_row[col_idx + 7];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v, vl0); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v, vl0); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v, vl0); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v, vl0); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval4, v, vl0); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval5, v, vl0); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval6, v, vl0); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vl0); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval7, v, vl0); }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m1(rem1);
            int oc1 = oc + (int)vl0;
            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[oc], vl0);
            vint32m2_t vacc1 = __riscv_vle32_v_i32m2(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt40 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt50 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt60 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt70 = &weight_buffer[(col_idx + 7) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;
                const int16_t *wt41 = wt40 + vl0;
                const int16_t *wt51 = wt50 + vl0;
                const int16_t *wt61 = wt60 + vl0;
                const int16_t *wt71 = wt70 + vl0;

                if (inval0 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt00, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt01, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, v1, vl1); }
                if (inval1 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt10, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt11, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, v1, vl1); }
                if (inval2 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt20, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt21, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, v1, vl1); }
                if (inval3 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt30, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt31, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, v1, vl1); }
                if (inval4 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt40, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt41, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval4, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval4, v1, vl1); }
                if (inval5 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt50, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt51, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval5, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval5, v1, vl1); }
                if (inval6 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt60, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt61, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval6, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval6, v1, vl1); }
                if (inval7 != 0) { vint16m1_t v0 = __riscv_vle16_v_i16m1(wt70, vl0); vint16m1_t v1 = __riscv_vle16_v_i16m1(wt71, vl1); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval7, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval7, v1, vl1); }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m1_t vwt0 = __riscv_vle16_v_i16m1(wt0, vl0);
                vint16m1_t vwt1 = __riscv_vle16_v_i16m1(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc4_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc4_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m1(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[bases[0]], vls[0]);
            vint32m2_t vacc1, vacc2, vacc3;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m2(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m2(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m2(&bias_i32[bases[3]], vls[3]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;

                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval7, v, vls[3]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[0]);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[1]);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[2]);
                    vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[3]);
                    vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, v, vls[3]);
                }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc8_m2(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc8_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m1(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m2_t vacc0 = __riscv_vle32_v_i32m2(&bias_i32[bases[0]], vls[0]);
            vint32m2_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m2(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m2(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m2(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m2(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m2(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m2(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m2(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;

                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval7, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[4]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[4]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[4]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[4]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval3, v, vls[4]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval4, v, vls[4]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval5, v, vls[4]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval6, v, vls[4]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval7, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[5]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[5]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[5]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[5]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval3, v, vls[5]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval4, v, vls[5]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval5, v, vls[5]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval6, v, vls[5]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval7, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[6]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[6]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[6]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[6]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval3, v, vls[6]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval4, v, vls[6]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval5, v, vls[6]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval6, v, vls[6]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval7, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[7]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[7]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[7]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[7]];
                    if (inval0 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval3, v, vls[7]); }
                    if (inval4 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt4, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval4, v, vls[7]); }
                    if (inval5 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt5, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval5, v, vls[7]); }
                    if (inval6 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt6, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval6, v, vls[7]); }
                    if (inval7 != 0) { vint16m1_t v = __riscv_vle16_v_i16m1(wt7, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval7, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[0]);
                    vacc0 = __riscv_vwmacc_vx_i32m2(vacc0, inval, v, vls[0]);
                }
                if (blocks > 1) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[1]);
                    vacc1 = __riscv_vwmacc_vx_i32m2(vacc1, inval, v, vls[1]);
                }
                if (blocks > 2) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[2]);
                    vacc2 = __riscv_vwmacc_vx_i32m2(vacc2, inval, v, vls[2]);
                }
                if (blocks > 3) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[3]);
                    vacc3 = __riscv_vwmacc_vx_i32m2(vacc3, inval, v, vls[3]);
                }
                if (blocks > 4) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[4]);
                    vacc4 = __riscv_vwmacc_vx_i32m2(vacc4, inval, v, vls[4]);
                }
                if (blocks > 5) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[5]);
                    vacc5 = __riscv_vwmacc_vx_i32m2(vacc5, inval, v, vls[5]);
                }
                if (blocks > 6) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[6]);
                    vacc6 = __riscv_vwmacc_vx_i32m2(vacc6, inval, v, vls[6]);
                }
                if (blocks > 7) {
                    const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]];
                    vint16m1_t v = __riscv_vle16_v_i16m1(wt, vls[7]);
                    vacc7 = __riscv_vwmacc_vx_i32m2(vacc7, inval, v, vls[7]);
                }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_chaining2_m2(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];


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

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
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

void conv1d_i8_vpu_chaining4_m2(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);
            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0) {
                        continue;
                    }
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }
    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_chaining8_m2(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m2.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m2_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m2(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m1(outC - oc);
            vint32m2_t vacc = __riscv_vle32_v_i32m2(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];
                    int16_t inval4 = (int16_t)in_ptr[ic + 4];
                    int16_t inval5 = (int16_t)in_ptr[ic + 5];
                    int16_t inval6 = (int16_t)in_ptr[ic + 6];
                    int16_t inval7 = (int16_t)in_ptr[ic + 7];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const int16_t *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const int16_t *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const int16_t *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const int16_t *wt7 = &weight_buffer[col_idx7 * outC + oc];

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
                        if (inval4 != 0) {
                            vint16m1_t vwt4 = __riscv_vle16_v_i16m1(wt4, vl);
                            vacc = __riscv_vwmacc_vx_i32m2(vacc, inval4, vwt4, vl);
                        }
                        if (inval5 != 0) {
                            vint16m1_t vwt5 = __riscv_vle16_v_i16m1(wt5, vl);
                            vacc = __riscv_vwmacc_vx_i32m2(vacc, inval5, vwt5, vl);
                        }
                        if (inval6 != 0) {
                            vint16m1_t vwt6 = __riscv_vle16_v_i16m1(wt6, vl);
                            vacc = __riscv_vwmacc_vx_i32m2(vacc, inval6, vwt6, vl);
                        }
                        if (inval7 != 0) {
                            vint16m1_t vwt7 = __riscv_vle16_v_i16m1(wt7, vl);
                            vacc = __riscv_vwmacc_vx_i32m2(vacc, inval7, vwt7, vl);
                        }
                    }
                }

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m1_t vwt = __riscv_vle16_v_i16m1(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m2(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_m4(NNModule *layer, void *input, void *output)
{
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0)
                        continue;              
                    int col_idx = k * inC + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
                }
            }
            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            vint32m4_t vacc1 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                if (inval0 == 0 && inval1 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                if (inval0 != 0)
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt, vl);
                if (inval1 != 0)
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m4(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    int pos = 0;
    for (; pos + 3 < outW; pos += 4) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        const int8_t *row2 = &input_i8[(pos + 2) * cols];
        const int8_t *row3 = &input_i8[(pos + 3) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            vint32m4_t vacc1 = vacc0;
            vint32m4_t vacc2 = vacc0;
            vint32m4_t vacc3 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                int16_t inval2 = (int16_t)row2[col_idx];
                int16_t inval3 = (int16_t)row3[col_idx];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                if (inval0 != 0) vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt, vl);
                if (inval1 != 0) vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt, vl);
                if (inval2 != 0) vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval2, vwt, vl);
                if (inval3 != 0) vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval3, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m4(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    int pos = 0;
    for (; pos + 7 < outW; pos += 8) {
        const int8_t *row[8] = {
            &input_i8[(pos + 0) * cols], &input_i8[(pos + 1) * cols],
            &input_i8[(pos + 2) * cols], &input_i8[(pos + 3) * cols],
            &input_i8[(pos + 4) * cols], &input_i8[(pos + 5) * cols],
            &input_i8[(pos + 6) * cols], &input_i8[(pos + 7) * cols],
        };
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            vint32m4_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            vint32m4_t vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval[8] = {
                    (int16_t)row[0][col_idx], (int16_t)row[1][col_idx],
                    (int16_t)row[2][col_idx], (int16_t)row[3][col_idx],
                    (int16_t)row[4][col_idx], (int16_t)row[5][col_idx],
                    (int16_t)row[6][col_idx], (int16_t)row[7][col_idx],
                };
                if (inval[0] == 0 && inval[1] == 0 && inval[2] == 0 && inval[3] == 0 &&
                    inval[4] == 0 && inval[5] == 0 && inval[6] == 0 && inval[7] == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                if (inval[0] != 0) vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval[0], vwt, vl);
                if (inval[1] != 0) vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval[1], vwt, vl);
                if (inval[2] != 0) vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval[2], vwt, vl);
                if (inval[3] != 0) vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval[3], vwt, vl);
                if (inval[4] != 0) vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval[4], vwt, vl);
                if (inval[5] != 0) vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval[5], vwt, vl);
                if (inval[6] != 0) vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval[6], vwt, vl);
                if (inval[7] != 0) vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval[7], vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[(pos + 0) * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &M[oc], &Z[oc], &output_i8[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &M[oc], &Z[oc], &output_i8[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &M[oc], &Z[oc], &output_i8[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &M[oc], &Z[oc], &output_i8[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m4(layer, input, output);

    safe_free(im2col_input);
}

static void conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(NNModule *layer, void *input, void *output,
                                                           int unroll_cols, int acc2,
                                                           const char *kernel_name)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\n", kernel_name);
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];

        if (!acc2) {
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e16m2(outC - oc);
                vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
                vint32m4_t vacc1 = vacc0;

                for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                    int block = cols - col_idx;
                    if (block > unroll_cols)
                        block = unroll_cols;
                    for (int u = 0; u < block; ++u) {
                        int k = col_idx + u;
                        int16_t inval0 = (int16_t)row0[k];
                        int16_t inval1 = (int16_t)row1[k];
                        if (inval0 == 0 && inval1 == 0)
                            continue;
                        const int16_t *wt = &weight_buffer[k * outC + oc];
                        vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                        if (inval0 != 0)
                            vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt, vl);
                        if (inval1 != 0)
                            vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt, vl);
                    }
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
                act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
                oc += (int)vl;
            }
            continue;
        }

        int oc = 0;
        while (oc < outC) {
            int base0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m2(outC - base0);
            int base1 = base0 + (int)vl0;
            int blocks = 1;
            size_t vl1 = 0;
            if (base1 < outC) {
                vl1 = __riscv_vsetvl_e16m2(outC - base1);
                blocks = 2;
            }

            vint32m4_t vacc00 = __riscv_vle32_v_i32m4(&bias_i32[base0], vl0);
            vint32m4_t vacc01 = vacc00;
            vint32m4_t vacc10 = vacc00;
            vint32m4_t vacc11 = vacc00;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_i32m4(&bias_i32[base1], vl1);
                vacc11 = vacc10;
            }

            for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                int block = cols - col_idx;
                if (block > unroll_cols)
                    block = unroll_cols;
                for (int u = 0; u < block; ++u) {
                    int k = col_idx + u;
                    int16_t inval0 = (int16_t)row0[k];
                    int16_t inval1 = (int16_t)row1[k];
                    if (inval0 == 0 && inval1 == 0)
                        continue;
                    const int16_t *wt0 = &weight_buffer[k * outC + base0];
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                    if (inval0 != 0)
                        vacc00 = __riscv_vwmacc_vx_i32m4(vacc00, inval0, vwt0, vl0);
                    if (inval1 != 0)
                        vacc01 = __riscv_vwmacc_vx_i32m4(vacc01, inval1, vwt0, vl0);
                    if (blocks > 1) {
                        const int16_t *wt1 = &weight_buffer[k * outC + base1];
                        vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl1);
                        if (inval0 != 0)
                            vacc10 = __riscv_vwmacc_vx_i32m4(vacc10, inval0, vwt1, vl1);
                        if (inval1 != 0)
                            vacc11 = __riscv_vwmacc_vx_i32m4(vacc11, inval1, vwt1, vl1);
                    }
                }
            }

            act_kernel(vacc00, &M[base0], &Z[base0], &output_i8[pos * outC + base0], vl0);
            act_kernel(vacc01, &M[base0], &Z[base0], &output_i8[(pos + 1) * outC + base0], vl0);
            if (blocks > 1) {
                act_kernel(vacc10, &M[base1], &Z[base1], &output_i8[pos * outC + base1], vl1);
                act_kernel(vacc11, &M[base1], &Z[base1], &output_i8[(pos + 1) * outC + base1], vl1);
            }
            oc = base1 + (int)vl1;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m4(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 2, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m4");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 2, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m4");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 4, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m4");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 4, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m4");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 8, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m4");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m4(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m4_impl(layer, input, output, 8, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m4");
}

void conv1d_i8_vpu_im2col_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    if (inval0 != 0 && inval1 != 0) {
                        vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                        vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt1, vl0);
                    } else {
                        if (inval0 != 0) { vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt1, vl0); }
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt, vl0);
                }
                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m2(rem1);
            int oc1 = oc + (int)vl0;
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
            vint32m4_t vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;

                if (inval0 != 0 && inval1 != 0) {
                    vint16m2_t vwt00 = __riscv_vle16_v_i16m2(wt00, vl0);
                    vint16m2_t vwt10 = __riscv_vle16_v_i16m2(wt10, vl0);
                    vint16m2_t vwt01 = __riscv_vle16_v_i16m2(wt01, vl1);
                    vint16m2_t vwt11 = __riscv_vle16_v_i16m2(wt11, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt10, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt11, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m2_t vwt00 = __riscv_vle16_v_i16m2(wt00, vl0);
                        vint16m2_t vwt01 = __riscv_vle16_v_i16m2(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m2_t vwt10 = __riscv_vle16_v_i16m2(wt10, vl0);
                        vint16m2_t vwt11 = __riscv_vle16_v_i16m2(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt11, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc);
            int oc0 = oc;
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0;
            vint32m4_t vacc1, vacc2, vacc3;

            if (oc_cursor < outC) {
                groups = 2;
                oc1 = oc_cursor;
                vl1 = __riscv_vsetvl_e16m2(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1);
                oc_cursor = oc1 + (int)vl1;
            }
            if (oc_cursor < outC) {
                groups = 3;
                oc2 = oc_cursor;
                vl2 = __riscv_vsetvl_e16m2(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m4(&bias_i32[oc2], vl2);
                oc_cursor = oc2 + (int)vl2;
            }
            if (oc_cursor < outC) {
                groups = 4;
                oc3 = oc_cursor;
                vl3 = __riscv_vsetvl_e16m2(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m4(&bias_i32[oc3], vl3);
                oc_cursor = oc3 + (int)vl3;
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base0 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, vwt3, vl3); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base1 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, vwt3, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base + oc0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, vwt3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc);
            int oc0 = oc;
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0, vl4 = 0, vl5 = 0, vl6 = 0, vl7 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0, oc4 = 0, oc5 = 0, oc6 = 0, oc7 = 0;
            vint32m4_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;

            if (oc_cursor < outC) { groups = 2; oc1 = oc_cursor; vl1 = __riscv_vsetvl_e16m2(outC - oc1); vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1); oc_cursor = oc1 + (int)vl1; }
            if (oc_cursor < outC) { groups = 3; oc2 = oc_cursor; vl2 = __riscv_vsetvl_e16m2(outC - oc2); vacc2 = __riscv_vle32_v_i32m4(&bias_i32[oc2], vl2); oc_cursor = oc2 + (int)vl2; }
            if (oc_cursor < outC) { groups = 4; oc3 = oc_cursor; vl3 = __riscv_vsetvl_e16m2(outC - oc3); vacc3 = __riscv_vle32_v_i32m4(&bias_i32[oc3], vl3); oc_cursor = oc3 + (int)vl3; }
            if (oc_cursor < outC) { groups = 5; oc4 = oc_cursor; vl4 = __riscv_vsetvl_e16m2(outC - oc4); vacc4 = __riscv_vle32_v_i32m4(&bias_i32[oc4], vl4); oc_cursor = oc4 + (int)vl4; }
            if (oc_cursor < outC) { groups = 6; oc5 = oc_cursor; vl5 = __riscv_vsetvl_e16m2(outC - oc5); vacc5 = __riscv_vle32_v_i32m4(&bias_i32[oc5], vl5); oc_cursor = oc5 + (int)vl5; }
            if (oc_cursor < outC) { groups = 7; oc6 = oc_cursor; vl6 = __riscv_vsetvl_e16m2(outC - oc6); vacc6 = __riscv_vle32_v_i32m4(&bias_i32[oc6], vl6); oc_cursor = oc6 + (int)vl6; }
            if (oc_cursor < outC) { groups = 8; oc7 = oc_cursor; vl7 = __riscv_vsetvl_e16m2(outC - oc7); vacc7 = __riscv_vle32_v_i32m4(&bias_i32[oc7], vl7); oc_cursor = oc7 + (int)vl7; }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base0 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, vwt3, vl3); }
                    if (groups > 4) { vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt_base0 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval0, vwt4, vl4); }
                    if (groups > 5) { vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt_base0 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval0, vwt5, vl5); }
                    if (groups > 6) { vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt_base0 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval0, vwt6, vl6); }
                    if (groups > 7) { vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt_base0 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval0, vwt7, vl7); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base1 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, vwt3, vl3); }
                    if (groups > 4) { vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt_base1 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval1, vwt4, vl4); }
                    if (groups > 5) { vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt_base1 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval1, vwt5, vl5); }
                    if (groups > 6) { vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt_base1 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval1, vwt6, vl6); }
                    if (groups > 7) { vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt_base1 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval1, vwt7, vl7); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt_base + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, vwt3, vl3); }
                if (groups > 4) { vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt_base + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval, vwt4, vl4); }
                if (groups > 5) { vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt_base + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval, vwt5, vl5); }
                if (groups > 6) { vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt_base + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval, vwt6, vl6); }
                if (groups > 7) { vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt_base + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval, vwt7, vl7); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            if (groups > 4) act_kernel(vacc4, &M[oc4], &Z[oc4], &output_i8[pos * outC + oc4], vl4);
            if (groups > 5) act_kernel(vacc5, &M[oc5], &Z[oc5], &output_i8[pos * outC + oc5], vl5);
            if (groups > 6) act_kernel(vacc6, &M[oc6], &Z[oc6], &output_i8[pos * outC + oc6], vl6);
            if (groups > 7) act_kernel(vacc7, &M[oc7], &Z[oc7], &output_i8[pos * outC + oc7], vl7);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 3 < cols; col_idx += 4) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];

                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

                    if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                        vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                        vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl0);
                        vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl0);
                        vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, vwt2, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, vwt3, vl0);
                    } else {
                        if (inval0 != 0) { vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt1, vl0); }
                        if (inval2 != 0) { vint16m2_t vwt2 = __riscv_vle16_v_i16m2(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, vwt2, vl0); }
                        if (inval3 != 0) { vint16m2_t vwt3 = __riscv_vle16_v_i16m2(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, vwt3, vl0); }
                    }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m2(rem1);
            int oc1 = oc + (int)vl0;
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
            vint32m4_t vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;

                if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                    vint16m2_t vwt00 = __riscv_vle16_v_i16m2(wt00, vl0);
                    vint16m2_t vwt10 = __riscv_vle16_v_i16m2(wt10, vl0);
                    vint16m2_t vwt20 = __riscv_vle16_v_i16m2(wt20, vl0);
                    vint16m2_t vwt30 = __riscv_vle16_v_i16m2(wt30, vl0);
                    vint16m2_t vwt01 = __riscv_vle16_v_i16m2(wt01, vl1);
                    vint16m2_t vwt11 = __riscv_vle16_v_i16m2(wt11, vl1);
                    vint16m2_t vwt21 = __riscv_vle16_v_i16m2(wt21, vl1);
                    vint16m2_t vwt31 = __riscv_vle16_v_i16m2(wt31, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt10, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, vwt20, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, vwt30, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt11, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, vwt21, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, vwt31, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m2_t vwt00 = __riscv_vle16_v_i16m2(wt00, vl0);
                        vint16m2_t vwt01 = __riscv_vle16_v_i16m2(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m2_t vwt10 = __riscv_vle16_v_i16m2(wt10, vl0);
                        vint16m2_t vwt11 = __riscv_vle16_v_i16m2(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, vwt11, vl1);
                    }
                    if (inval2 != 0) {
                        vint16m2_t vwt20 = __riscv_vle16_v_i16m2(wt20, vl0);
                        vint16m2_t vwt21 = __riscv_vle16_v_i16m2(wt21, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, vwt20, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, vwt21, vl1);
                    }
                    if (inval3 != 0) {
                        vint16m2_t vwt30 = __riscv_vle16_v_i16m2(wt30, vl0);
                        vint16m2_t vwt31 = __riscv_vle16_v_i16m2(wt31, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, vwt30, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, vwt31, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int oc0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc0);
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc0], vl0);

            int oc1 = oc0 + (int)vl0;
            size_t vl1 = 0;
            vint32m4_t vacc1;
            int have1 = oc1 < outC;
            if (have1) {
                vl1 = __riscv_vsetvl_e16m2(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1);
            }

            int oc2 = oc1 + (int)vl1;
            size_t vl2 = 0;
            vint32m4_t vacc2;
            int have2 = oc2 < outC;
            if (have2) {
                vl2 = __riscv_vsetvl_e16m2(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m4(&bias_i32[oc2], vl2);
            }

            int oc3 = oc2 + (int)vl2;
            size_t vl3 = 0;
            vint32m4_t vacc3;
            int have3 = oc3 < outC;
            if (have3) {
                vl3 = __riscv_vsetvl_e16m2(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m4(&bias_i32[oc3], vl3);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc0];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc0];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc0];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc0];
                if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt00, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v, vl0); }
                if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt10, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v, vl0); }
                if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt20, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v, vl0); }
                if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt30, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v, vl0); }

                if (have1) {
                    const int16_t *wt01 = &weight_buffer[(col_idx + 0) * outC + oc1];
                    const int16_t *wt11 = &weight_buffer[(col_idx + 1) * outC + oc1];
                    const int16_t *wt21 = &weight_buffer[(col_idx + 2) * outC + oc1];
                    const int16_t *wt31 = &weight_buffer[(col_idx + 3) * outC + oc1];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt01, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, v, vl1); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt11, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, v, vl1); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt21, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, v, vl1); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt31, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, v, vl1); }
                }
                if (have2) {
                    const int16_t *wt02 = &weight_buffer[(col_idx + 0) * outC + oc2];
                    const int16_t *wt12 = &weight_buffer[(col_idx + 1) * outC + oc2];
                    const int16_t *wt22 = &weight_buffer[(col_idx + 2) * outC + oc2];
                    const int16_t *wt32 = &weight_buffer[(col_idx + 3) * outC + oc2];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt02, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, v, vl2); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt12, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, v, vl2); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt22, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval2, v, vl2); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt32, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval3, v, vl2); }
                }
                if (have3) {
                    const int16_t *wt03 = &weight_buffer[(col_idx + 0) * outC + oc3];
                    const int16_t *wt13 = &weight_buffer[(col_idx + 1) * outC + oc3];
                    const int16_t *wt23 = &weight_buffer[(col_idx + 2) * outC + oc3];
                    const int16_t *wt33 = &weight_buffer[(col_idx + 3) * outC + oc3];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt03, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, v, vl3); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt13, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, v, vl3); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt23, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval2, v, vl3); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt33, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval3, v, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc0];
                vint16m2_t v0 = __riscv_vle16_v_i16m2(wt0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, v0, vl0);
                if (have1) { const int16_t *wt1 = &weight_buffer[col_idx * outC + oc1]; vint16m2_t v1 = __riscv_vle16_v_i16m2(wt1, vl1); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, v1, vl1); }
                if (have2) { const int16_t *wt2 = &weight_buffer[col_idx * outC + oc2]; vint16m2_t v2 = __riscv_vle16_v_i16m2(wt2, vl2); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, v2, vl2); }
                if (have3) { const int16_t *wt3 = &weight_buffer[col_idx * outC + oc3]; vint16m2_t v3 = __riscv_vle16_v_i16m2(wt3, vl3); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, v3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (have1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (have2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (have3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc3 + (int)vl3;
            if (!have3)
                oc = have2 ? (oc2 + (int)vl2) : (have1 ? (oc1 + (int)vl1) : (oc0 + (int)vl0));
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8];
            size_t vls[8];
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[bases[0]], vls[0]);
            vint32m4_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m4(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m4(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m4(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m4(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m4(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m4(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m4(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval3, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval3, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval3, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval3, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval3, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval3, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, v, vls[3]); }
                if (blocks > 4) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval, v, vls[4]); }
                if (blocks > 5) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval, v, vls[5]); }
                if (blocks > 6) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval, v, vls[6]); }
                if (blocks > 7) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval, v, vls[7]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

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

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m2_t vwt16 = __riscv_vle16_v_i16m2(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc2_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc2_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m2(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 7 < cols; col_idx += 8) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];
                    int16_t inval4 = (int16_t)input_row[col_idx + 4];
                    int16_t inval5 = (int16_t)input_row[col_idx + 5];
                    int16_t inval6 = (int16_t)input_row[col_idx + 6];
                    int16_t inval7 = (int16_t)input_row[col_idx + 7];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v, vl0); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v, vl0); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v, vl0); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v, vl0); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval4, v, vl0); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval5, v, vl0); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval6, v, vl0); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vl0); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval7, v, vl0); }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m2(rem1);
            int oc1 = oc + (int)vl0;
            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[oc], vl0);
            vint32m4_t vacc1 = __riscv_vle32_v_i32m4(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt40 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt50 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt60 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt70 = &weight_buffer[(col_idx + 7) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;
                const int16_t *wt41 = wt40 + vl0;
                const int16_t *wt51 = wt50 + vl0;
                const int16_t *wt61 = wt60 + vl0;
                const int16_t *wt71 = wt70 + vl0;

                if (inval0 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt00, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt01, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, v1, vl1); }
                if (inval1 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt10, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt11, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, v1, vl1); }
                if (inval2 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt20, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt21, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, v1, vl1); }
                if (inval3 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt30, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt31, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, v1, vl1); }
                if (inval4 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt40, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt41, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval4, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval4, v1, vl1); }
                if (inval5 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt50, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt51, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval5, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval5, v1, vl1); }
                if (inval6 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt60, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt61, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval6, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval6, v1, vl1); }
                if (inval7 != 0) { vint16m2_t v0 = __riscv_vle16_v_i16m2(wt70, vl0); vint16m2_t v1 = __riscv_vle16_v_i16m2(wt71, vl1); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval7, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval7, v1, vl1); }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m2_t vwt0 = __riscv_vle16_v_i16m2(wt0, vl0);
                vint16m2_t vwt1 = __riscv_vle16_v_i16m2(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc4_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc4_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[bases[0]], vls[0]);
            vint32m4_t vacc1, vacc2, vacc3;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m4(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m4(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m4(&bias_i32[bases[3]], vls[3]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval7, v, vls[3]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, v, vls[3]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc8_m4(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc8_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m2(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m4_t vacc0 = __riscv_vle32_v_i32m4(&bias_i32[bases[0]], vls[0]);
            vint32m4_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m4(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m4(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m4(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m4(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m4(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m4(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m4(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval7, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[4]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[4]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[4]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[4]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval3, v, vls[4]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval4, v, vls[4]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval5, v, vls[4]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval6, v, vls[4]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval7, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[5]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[5]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[5]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[5]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval3, v, vls[5]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval4, v, vls[5]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval5, v, vls[5]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval6, v, vls[5]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval7, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[6]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[6]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[6]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[6]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval3, v, vls[6]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval4, v, vls[6]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval5, v, vls[6]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval6, v, vls[6]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval7, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[7]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[7]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[7]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[7]];
                    if (inval0 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval3, v, vls[7]); }
                    if (inval4 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt4, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval4, v, vls[7]); }
                    if (inval5 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt5, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval5, v, vls[7]); }
                    if (inval6 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt6, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval6, v, vls[7]); }
                    if (inval7 != 0) { vint16m2_t v = __riscv_vle16_v_i16m2(wt7, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval7, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m4(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m4(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m4(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m4(vacc3, inval, v, vls[3]); }
                if (blocks > 4) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m4(vacc4, inval, v, vls[4]); }
                if (blocks > 5) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m4(vacc5, inval, v, vls[5]); }
                if (blocks > 6) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m4(vacc6, inval, v, vls[6]); }
                if (blocks > 7) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]]; vint16m2_t v = __riscv_vle16_v_i16m2(wt, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m4(vacc7, inval, v, vls[7]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_chaining2_m4(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];


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

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_chaining4_m4(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);
            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];

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

                for (; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0) {
                        continue;
                    }
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }
    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_chaining8_m4(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m4.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m4_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m4(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m2(outC - oc);
            vint32m4_t vacc = __riscv_vle32_v_i32m4(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];
                    int16_t inval4 = (int16_t)in_ptr[ic + 4];
                    int16_t inval5 = (int16_t)in_ptr[ic + 5];
                    int16_t inval6 = (int16_t)in_ptr[ic + 6];
                    int16_t inval7 = (int16_t)in_ptr[ic + 7];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const int16_t *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const int16_t *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const int16_t *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const int16_t *wt7 = &weight_buffer[col_idx7 * outC + oc];

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
                        if (inval4 != 0) {
                            vint16m2_t vwt4 = __riscv_vle16_v_i16m2(wt4, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc, inval4, vwt4, vl);
                        }
                        if (inval5 != 0) {
                            vint16m2_t vwt5 = __riscv_vle16_v_i16m2(wt5, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc, inval5, vwt5, vl);
                        }
                        if (inval6 != 0) {
                            vint16m2_t vwt6 = __riscv_vle16_v_i16m2(wt6, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc, inval6, vwt6, vl);
                        }
                        if (inval7 != 0) {
                            vint16m2_t vwt7 = __riscv_vle16_v_i16m2(wt7, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc, inval7, vwt7, vl);
                        }
                    }
                }

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m2_t vwt = __riscv_vle16_v_i16m2(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m4(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_m8(NNModule *layer, void *input, void *output)
{
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                for (int ic = 0; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0)
                        continue;              
                    int col_idx = k * inC + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
                }
            }
            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    if (layer->params.conv.padding > 0)
        safe_free(padded_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            vint32m8_t vacc1 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                if (inval0 == 0 && inval1 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                if (inval0 != 0)
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt, vl);
                if (inval1 != 0)
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m8(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    int pos = 0;
    for (; pos + 3 < outW; pos += 4) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];
        const int8_t *row2 = &input_i8[(pos + 2) * cols];
        const int8_t *row3 = &input_i8[(pos + 3) * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            vint32m8_t vacc1 = vacc0;
            vint32m8_t vacc2 = vacc0;
            vint32m8_t vacc3 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval0 = (int16_t)row0[col_idx];
                int16_t inval1 = (int16_t)row1[col_idx];
                int16_t inval2 = (int16_t)row2[col_idx];
                int16_t inval3 = (int16_t)row3[col_idx];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                if (inval0 != 0) vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt, vl);
                if (inval1 != 0) vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt, vl);
                if (inval2 != 0) vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval2, vwt, vl);
                if (inval3 != 0) vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval3, vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m8(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_reuse_w_8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;
    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    int pos = 0;
    for (; pos + 7 < outW; pos += 8) {
        const int8_t *row[8] = {
            &input_i8[(pos + 0) * cols], &input_i8[(pos + 1) * cols],
            &input_i8[(pos + 2) * cols], &input_i8[(pos + 3) * cols],
            &input_i8[(pos + 4) * cols], &input_i8[(pos + 5) * cols],
            &input_i8[(pos + 6) * cols], &input_i8[(pos + 7) * cols],
        };
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            vint32m8_t vacc1 = vacc0, vacc2 = vacc0, vacc3 = vacc0;
            vint32m8_t vacc4 = vacc0, vacc5 = vacc0, vacc6 = vacc0, vacc7 = vacc0;
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval[8] = {
                    (int16_t)row[0][col_idx], (int16_t)row[1][col_idx],
                    (int16_t)row[2][col_idx], (int16_t)row[3][col_idx],
                    (int16_t)row[4][col_idx], (int16_t)row[5][col_idx],
                    (int16_t)row[6][col_idx], (int16_t)row[7][col_idx],
                };
                if (inval[0] == 0 && inval[1] == 0 && inval[2] == 0 && inval[3] == 0 &&
                    inval[4] == 0 && inval[5] == 0 && inval[6] == 0 && inval[7] == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                if (inval[0] != 0) vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval[0], vwt, vl);
                if (inval[1] != 0) vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval[1], vwt, vl);
                if (inval[2] != 0) vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval[2], vwt, vl);
                if (inval[3] != 0) vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval[3], vwt, vl);
                if (inval[4] != 0) vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval[4], vwt, vl);
                if (inval[5] != 0) vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval[5], vwt, vl);
                if (inval[6] != 0) vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval[6], vwt, vl);
                if (inval[7] != 0) vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval[7], vwt, vl);
            }
            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[(pos + 0) * outC + oc], vl);
            act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
            act_kernel(vacc2, &M[oc], &Z[oc], &output_i8[(pos + 2) * outC + oc], vl);
            act_kernel(vacc3, &M[oc], &Z[oc], &output_i8[(pos + 3) * outC + oc], vl);
            act_kernel(vacc4, &M[oc], &Z[oc], &output_i8[(pos + 4) * outC + oc], vl);
            act_kernel(vacc5, &M[oc], &Z[oc], &output_i8[(pos + 5) * outC + oc], vl);
            act_kernel(vacc6, &M[oc], &Z[oc], &output_i8[(pos + 6) * outC + oc], vl);
            act_kernel(vacc7, &M[oc], &Z[oc], &output_i8[(pos + 7) * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m8(layer, input, output);

    safe_free(im2col_input);
}

static void conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(NNModule *layer, void *input, void *output,
                                                           int unroll_cols, int acc2,
                                                           const char *kernel_name)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in %s.\n", kernel_name);
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    int pos = 0;
    for (; pos + 1 < outW; pos += 2) {
        const int8_t *row0 = &input_i8[pos * cols];
        const int8_t *row1 = &input_i8[(pos + 1) * cols];

        if (!acc2) {
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e16m4(outC - oc);
                vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
                vint32m8_t vacc1 = vacc0;

                for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                    int block = cols - col_idx;
                    if (block > unroll_cols)
                        block = unroll_cols;
                    for (int u = 0; u < block; ++u) {
                        int k = col_idx + u;
                        int16_t inval0 = (int16_t)row0[k];
                        int16_t inval1 = (int16_t)row1[k];
                        if (inval0 == 0 && inval1 == 0)
                            continue;
                        const int16_t *wt = &weight_buffer[k * outC + oc];
                        vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                        if (inval0 != 0)
                            vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt, vl);
                        if (inval1 != 0)
                            vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt, vl);
                    }
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
                act_kernel(vacc1, &M[oc], &Z[oc], &output_i8[(pos + 1) * outC + oc], vl);
                oc += (int)vl;
            }
            continue;
        }

        int oc = 0;
        while (oc < outC) {
            int base0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m4(outC - base0);
            int base1 = base0 + (int)vl0;
            int blocks = 1;
            size_t vl1 = 0;
            if (base1 < outC) {
                vl1 = __riscv_vsetvl_e16m4(outC - base1);
                blocks = 2;
            }

            vint32m8_t vacc00 = __riscv_vle32_v_i32m8(&bias_i32[base0], vl0);
            vint32m8_t vacc01 = vacc00;
            vint32m8_t vacc10 = vacc00;
            vint32m8_t vacc11 = vacc00;
            if (blocks > 1) {
                vacc10 = __riscv_vle32_v_i32m8(&bias_i32[base1], vl1);
                vacc11 = vacc10;
            }

            for (int col_idx = 0; col_idx < cols; col_idx += unroll_cols) {
                int block = cols - col_idx;
                if (block > unroll_cols)
                    block = unroll_cols;
                for (int u = 0; u < block; ++u) {
                    int k = col_idx + u;
                    int16_t inval0 = (int16_t)row0[k];
                    int16_t inval1 = (int16_t)row1[k];
                    if (inval0 == 0 && inval1 == 0)
                        continue;
                    const int16_t *wt0 = &weight_buffer[k * outC + base0];
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                    if (inval0 != 0)
                        vacc00 = __riscv_vwmacc_vx_i32m8(vacc00, inval0, vwt0, vl0);
                    if (inval1 != 0)
                        vacc01 = __riscv_vwmacc_vx_i32m8(vacc01, inval1, vwt0, vl0);
                    if (blocks > 1) {
                        const int16_t *wt1 = &weight_buffer[k * outC + base1];
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl1);
                        if (inval0 != 0)
                            vacc10 = __riscv_vwmacc_vx_i32m8(vacc10, inval0, vwt1, vl1);
                        if (inval1 != 0)
                            vacc11 = __riscv_vwmacc_vx_i32m8(vacc11, inval1, vwt1, vl1);
                    }
                }
            }

            act_kernel(vacc00, &M[base0], &Z[base0], &output_i8[pos * outC + base0], vl0);
            act_kernel(vacc01, &M[base0], &Z[base0], &output_i8[(pos + 1) * outC + base0], vl0);
            if (blocks > 1) {
                act_kernel(vacc10, &M[base1], &Z[base1], &output_i8[pos * outC + base1], vl1);
                act_kernel(vacc11, &M[base1], &Z[base1], &output_i8[(pos + 1) * outC + base1], vl1);
            }
            oc = base1 + (int)vl1;
        }
    }

    if (pos < outW)
        conv1d_i8_vpu_im2col_m8(layer, input, output);

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 2, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_m8");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 2, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll2_acc2_m8");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 4, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_m8");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 4, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll4_acc2_m8");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 8, 0,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_m8");
}

void conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m8(NNModule *layer, void *input, void *output)
{
    conv1d_i8_vpu_im2col_reuse_w_2_unroll_m8_impl(layer, input, output, 8, 1,
                                                   "conv1d_i8_vpu_im2col_reuse_w_2_unroll8_acc2_m8");
}

void conv1d_i8_vpu_im2col_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            for (int col_idx = 0; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];

                if (inval0 != 0 && inval1 != 0) {
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                    vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                } else {
                    if (inval0 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                    }
                    if (inval1 != 0) {
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
                int col_idx = 0;
                for (; col_idx + 1 < cols; col_idx += 2) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    if (inval0 != 0 && inval1 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt1, vl0);
                    } else {
                        if (inval0 != 0) { vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt1, vl0); }
                    }
                }
                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt, vl0);
                }
                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m4(rem1);
            int oc1 = oc + (int)vl0;
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
            vint32m8_t vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;

                if (inval0 != 0 && inval1 != 0) {
                    vint16m4_t vwt00 = __riscv_vle16_v_i16m4(wt00, vl0);
                    vint16m4_t vwt10 = __riscv_vle16_v_i16m4(wt10, vl0);
                    vint16m4_t vwt01 = __riscv_vle16_v_i16m4(wt01, vl1);
                    vint16m4_t vwt11 = __riscv_vle16_v_i16m4(wt11, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt10, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt11, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m4_t vwt00 = __riscv_vle16_v_i16m4(wt00, vl0);
                        vint16m4_t vwt01 = __riscv_vle16_v_i16m4(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m4_t vwt10 = __riscv_vle16_v_i16m4(wt10, vl0);
                        vint16m4_t vwt11 = __riscv_vle16_v_i16m4(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt11, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc);
            int oc0 = oc;
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0;
            vint32m8_t vacc1, vacc2, vacc3;

            if (oc_cursor < outC) {
                groups = 2;
                oc1 = oc_cursor;
                vl1 = __riscv_vsetvl_e16m4(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1);
                oc_cursor = oc1 + (int)vl1;
            }
            if (oc_cursor < outC) {
                groups = 3;
                oc2 = oc_cursor;
                vl2 = __riscv_vsetvl_e16m4(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m8(&bias_i32[oc2], vl2);
                oc_cursor = oc2 + (int)vl2;
            }
            if (oc_cursor < outC) {
                groups = 4;
                oc3 = oc_cursor;
                vl3 = __riscv_vsetvl_e16m4(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m8(&bias_i32[oc3], vl3);
                oc_cursor = oc3 + (int)vl3;
            }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base0 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, vwt3, vl3); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base1 + oc0, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, vwt3, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base + oc0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, vwt3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll2_acc8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll2_acc8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        while (oc < outC) {
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc);
            int oc0 = oc;
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc0], vl0);

            int groups = 1;
            int oc_cursor = oc0 + (int)vl0;
            size_t vl1 = 0, vl2 = 0, vl3 = 0, vl4 = 0, vl5 = 0, vl6 = 0, vl7 = 0;
            int oc1 = 0, oc2 = 0, oc3 = 0, oc4 = 0, oc5 = 0, oc6 = 0, oc7 = 0;
            vint32m8_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;

            if (oc_cursor < outC) { groups = 2; oc1 = oc_cursor; vl1 = __riscv_vsetvl_e16m4(outC - oc1); vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1); oc_cursor = oc1 + (int)vl1; }
            if (oc_cursor < outC) { groups = 3; oc2 = oc_cursor; vl2 = __riscv_vsetvl_e16m4(outC - oc2); vacc2 = __riscv_vle32_v_i32m8(&bias_i32[oc2], vl2); oc_cursor = oc2 + (int)vl2; }
            if (oc_cursor < outC) { groups = 4; oc3 = oc_cursor; vl3 = __riscv_vsetvl_e16m4(outC - oc3); vacc3 = __riscv_vle32_v_i32m8(&bias_i32[oc3], vl3); oc_cursor = oc3 + (int)vl3; }
            if (oc_cursor < outC) { groups = 5; oc4 = oc_cursor; vl4 = __riscv_vsetvl_e16m4(outC - oc4); vacc4 = __riscv_vle32_v_i32m8(&bias_i32[oc4], vl4); oc_cursor = oc4 + (int)vl4; }
            if (oc_cursor < outC) { groups = 6; oc5 = oc_cursor; vl5 = __riscv_vsetvl_e16m4(outC - oc5); vacc5 = __riscv_vle32_v_i32m8(&bias_i32[oc5], vl5); oc_cursor = oc5 + (int)vl5; }
            if (oc_cursor < outC) { groups = 7; oc6 = oc_cursor; vl6 = __riscv_vsetvl_e16m4(outC - oc6); vacc6 = __riscv_vle32_v_i32m8(&bias_i32[oc6], vl6); oc_cursor = oc6 + (int)vl6; }
            if (oc_cursor < outC) { groups = 8; oc7 = oc_cursor; vl7 = __riscv_vsetvl_e16m4(outC - oc7); vacc7 = __riscv_vle32_v_i32m8(&bias_i32[oc7], vl7); oc_cursor = oc7 + (int)vl7; }

            int col_idx = 0;
            for (; col_idx + 1 < cols; col_idx += 2) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];

                if (inval0 != 0) {
                    const int16_t *wt_base0 = &weight_buffer[(col_idx + 0) * outC];
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base0 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0);
                    if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base0 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt1, vl1); }
                    if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base0 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, vwt2, vl2); }
                    if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base0 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, vwt3, vl3); }
                    if (groups > 4) { vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt_base0 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval0, vwt4, vl4); }
                    if (groups > 5) { vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt_base0 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval0, vwt5, vl5); }
                    if (groups > 6) { vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt_base0 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval0, vwt6, vl6); }
                    if (groups > 7) { vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt_base0 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval0, vwt7, vl7); }
                }
                if (inval1 != 0) {
                    const int16_t *wt_base1 = &weight_buffer[(col_idx + 1) * outC];
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base1 + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt0, vl0);
                    if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base1 + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt1, vl1); }
                    if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base1 + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, vwt2, vl2); }
                    if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base1 + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, vwt3, vl3); }
                    if (groups > 4) { vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt_base1 + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval1, vwt4, vl4); }
                    if (groups > 5) { vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt_base1 + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval1, vwt5, vl5); }
                    if (groups > 6) { vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt_base1 + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval1, vwt6, vl6); }
                    if (groups > 7) { vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt_base1 + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval1, vwt7, vl7); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt_base = &weight_buffer[col_idx * outC];
                vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt_base + oc0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt0, vl0);
                if (groups > 1) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt_base + oc1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, vwt1, vl1); }
                if (groups > 2) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt_base + oc2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, vwt2, vl2); }
                if (groups > 3) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt_base + oc3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, vwt3, vl3); }
                if (groups > 4) { vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt_base + oc4, vl4); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval, vwt4, vl4); }
                if (groups > 5) { vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt_base + oc5, vl5); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval, vwt5, vl5); }
                if (groups > 6) { vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt_base + oc6, vl6); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval, vwt6, vl6); }
                if (groups > 7) { vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt_base + oc7, vl7); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval, vwt7, vl7); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (groups > 1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (groups > 2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (groups > 3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            if (groups > 4) act_kernel(vacc4, &M[oc4], &Z[oc4], &output_i8[pos * outC + oc4], vl4);
            if (groups > 5) act_kernel(vacc5, &M[oc5], &Z[oc5], &output_i8[pos * outC + oc5], vl5);
            if (groups > 6) act_kernel(vacc6, &M[oc6], &Z[oc6], &output_i8[pos * outC + oc6], vl6);
            if (groups > 7) act_kernel(vacc7, &M[oc7], &Z[oc7], &output_i8[pos * outC + oc7], vl7);
            oc = oc_cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

                if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                    vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                    vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                    vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                } else {
                    if (inval0 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                    }
                    if (inval1 != 0) {
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                    }
                    if (inval2 != 0) {
                        vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                    }
                    if (inval3 != 0) {
                        vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 3 < cols; col_idx += 4) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];

                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];

                    if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl0);
                        vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl0);
                        vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt1, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, vwt2, vl0);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, vwt3, vl0);
                    } else {
                        if (inval0 != 0) { vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt0, vl0); }
                        if (inval1 != 0) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt1, vl0); }
                        if (inval2 != 0) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, vwt2, vl0); }
                        if (inval3 != 0) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, vwt3, vl0); }
                    }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m4(rem1);
            int oc1 = oc + (int)vl0;
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
            vint32m8_t vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;

                if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                    vint16m4_t vwt00 = __riscv_vle16_v_i16m4(wt00, vl0);
                    vint16m4_t vwt10 = __riscv_vle16_v_i16m4(wt10, vl0);
                    vint16m4_t vwt20 = __riscv_vle16_v_i16m4(wt20, vl0);
                    vint16m4_t vwt30 = __riscv_vle16_v_i16m4(wt30, vl0);
                    vint16m4_t vwt01 = __riscv_vle16_v_i16m4(wt01, vl1);
                    vint16m4_t vwt11 = __riscv_vle16_v_i16m4(wt11, vl1);
                    vint16m4_t vwt21 = __riscv_vle16_v_i16m4(wt21, vl1);
                    vint16m4_t vwt31 = __riscv_vle16_v_i16m4(wt31, vl1);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt00, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt10, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, vwt20, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, vwt30, vl0);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt01, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt11, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, vwt21, vl1);
                    vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, vwt31, vl1);
                } else {
                    if (inval0 != 0) {
                        vint16m4_t vwt00 = __riscv_vle16_v_i16m4(wt00, vl0);
                        vint16m4_t vwt01 = __riscv_vle16_v_i16m4(wt01, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, vwt00, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, vwt01, vl1);
                    }
                    if (inval1 != 0) {
                        vint16m4_t vwt10 = __riscv_vle16_v_i16m4(wt10, vl0);
                        vint16m4_t vwt11 = __riscv_vle16_v_i16m4(wt11, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, vwt10, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, vwt11, vl1);
                    }
                    if (inval2 != 0) {
                        vint16m4_t vwt20 = __riscv_vle16_v_i16m4(wt20, vl0);
                        vint16m4_t vwt21 = __riscv_vle16_v_i16m4(wt21, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, vwt20, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, vwt21, vl1);
                    }
                    if (inval3 != 0) {
                        vint16m4_t vwt30 = __riscv_vle16_v_i16m4(wt30, vl0);
                        vint16m4_t vwt31 = __riscv_vle16_v_i16m4(wt31, vl1);
                        vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, vwt30, vl0);
                        vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, vwt31, vl1);
                    }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int oc0 = oc;
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc0);
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc0], vl0);

            int oc1 = oc0 + (int)vl0;
            size_t vl1 = 0;
            vint32m8_t vacc1;
            int have1 = oc1 < outC;
            if (have1) {
                vl1 = __riscv_vsetvl_e16m4(outC - oc1);
                vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1);
            }

            int oc2 = oc1 + (int)vl1;
            size_t vl2 = 0;
            vint32m8_t vacc2;
            int have2 = oc2 < outC;
            if (have2) {
                vl2 = __riscv_vsetvl_e16m4(outC - oc2);
                vacc2 = __riscv_vle32_v_i32m8(&bias_i32[oc2], vl2);
            }

            int oc3 = oc2 + (int)vl2;
            size_t vl3 = 0;
            vint32m8_t vacc3;
            int have3 = oc3 < outC;
            if (have3) {
                vl3 = __riscv_vsetvl_e16m4(outC - oc3);
                vacc3 = __riscv_vle32_v_i32m8(&bias_i32[oc3], vl3);
            }

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];

                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;

                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc0];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc0];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc0];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc0];
                if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt00, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v, vl0); }
                if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt10, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v, vl0); }
                if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt20, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v, vl0); }
                if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt30, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v, vl0); }

                if (have1) {
                    const int16_t *wt01 = &weight_buffer[(col_idx + 0) * outC + oc1];
                    const int16_t *wt11 = &weight_buffer[(col_idx + 1) * outC + oc1];
                    const int16_t *wt21 = &weight_buffer[(col_idx + 2) * outC + oc1];
                    const int16_t *wt31 = &weight_buffer[(col_idx + 3) * outC + oc1];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt01, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, v, vl1); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt11, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, v, vl1); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt21, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, v, vl1); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt31, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, v, vl1); }
                }
                if (have2) {
                    const int16_t *wt02 = &weight_buffer[(col_idx + 0) * outC + oc2];
                    const int16_t *wt12 = &weight_buffer[(col_idx + 1) * outC + oc2];
                    const int16_t *wt22 = &weight_buffer[(col_idx + 2) * outC + oc2];
                    const int16_t *wt32 = &weight_buffer[(col_idx + 3) * outC + oc2];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt02, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, v, vl2); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt12, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, v, vl2); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt22, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval2, v, vl2); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt32, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval3, v, vl2); }
                }
                if (have3) {
                    const int16_t *wt03 = &weight_buffer[(col_idx + 0) * outC + oc3];
                    const int16_t *wt13 = &weight_buffer[(col_idx + 1) * outC + oc3];
                    const int16_t *wt23 = &weight_buffer[(col_idx + 2) * outC + oc3];
                    const int16_t *wt33 = &weight_buffer[(col_idx + 3) * outC + oc3];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt03, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, v, vl3); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt13, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, v, vl3); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt23, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval2, v, vl3); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt33, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval3, v, vl3); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc0];
                vint16m4_t v0 = __riscv_vle16_v_i16m4(wt0, vl0);
                vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, v0, vl0);
                if (have1) { const int16_t *wt1 = &weight_buffer[col_idx * outC + oc1]; vint16m4_t v1 = __riscv_vle16_v_i16m4(wt1, vl1); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, v1, vl1); }
                if (have2) { const int16_t *wt2 = &weight_buffer[col_idx * outC + oc2]; vint16m4_t v2 = __riscv_vle16_v_i16m4(wt2, vl2); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, v2, vl2); }
                if (have3) { const int16_t *wt3 = &weight_buffer[col_idx * outC + oc3]; vint16m4_t v3 = __riscv_vle16_v_i16m4(wt3, vl3); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, v3, vl3); }
            }

            act_kernel(vacc0, &M[oc0], &Z[oc0], &output_i8[pos * outC + oc0], vl0);
            if (have1) act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            if (have2) act_kernel(vacc2, &M[oc2], &Z[oc2], &output_i8[pos * outC + oc2], vl2);
            if (have3) act_kernel(vacc3, &M[oc3], &Z[oc3], &output_i8[pos * outC + oc3], vl3);
            oc = oc3 + (int)vl3;
            if (!have3)
                oc = have2 ? (oc2 + (int)vl2) : (have1 ? (oc1 + (int)vl1) : (oc0 + (int)vl0));
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll4_acc8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll4_acc8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8];
            size_t vls[8];
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[bases[0]], vls[0]);
            vint32m8_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m8(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m8(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m8(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m8(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m8(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m8(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m8(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 3 < cols; col_idx += 4) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval3, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval3, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval3, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval3, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval3, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval3, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, v, vls[3]); }
                if (blocks > 4) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval, v, vls[4]); }
                if (blocks > 5) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval, v, vls[5]); }
                if (blocks > 6) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval, v, vls[6]); }
                if (blocks > 7) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval, v, vls[7]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];

                const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

                if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0 &&
                    inval4 != 0 && inval5 != 0 && inval6 != 0 && inval7 != 0) {
                    vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                    vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                    vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                    vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                    vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt4, vl);
                    vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt5, vl);
                    vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt6, vl);
                    vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt7, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt4, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt5, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt6, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt7, vl);
                } else {
                    if (inval0 != 0) { vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl); }
                    if (inval1 != 0) { vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl); }
                    if (inval2 != 0) { vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl); }
                    if (inval3 != 0) { vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl); }
                    if (inval4 != 0) { vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt4, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt4, vl); }
                    if (inval5 != 0) { vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt5, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt5, vl); }
                    if (inval6 != 0) { vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt6, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt6, vl); }
                    if (inval7 != 0) { vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt7, vl); vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt7, vl); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt16, vl);
            }
            act_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += vl;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc2_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc2_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;

        for (; oc < outC; ) {
            size_t vl0 = __riscv_vsetvl_e16m4(outC - oc);
            size_t rem1 = (size_t)(outC - oc) - vl0;

            if (rem1 == 0) {
                vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
                int col_idx = 0;

                for (; col_idx + 7 < cols; col_idx += 8) {
                    int16_t inval0 = (int16_t)input_row[col_idx + 0];
                    int16_t inval1 = (int16_t)input_row[col_idx + 1];
                    int16_t inval2 = (int16_t)input_row[col_idx + 2];
                    int16_t inval3 = (int16_t)input_row[col_idx + 3];
                    int16_t inval4 = (int16_t)input_row[col_idx + 4];
                    int16_t inval5 = (int16_t)input_row[col_idx + 5];
                    int16_t inval6 = (int16_t)input_row[col_idx + 6];
                    int16_t inval7 = (int16_t)input_row[col_idx + 7];
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + oc];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + oc];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + oc];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + oc];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + oc];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + oc];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + oc];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + oc];

                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v, vl0); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v, vl0); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v, vl0); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v, vl0); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval4, v, vl0); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval5, v, vl0); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval6, v, vl0); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vl0); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval7, v, vl0); }
                }

                for (; col_idx < cols; ++col_idx) {
                    int16_t inval = (int16_t)input_row[col_idx];
                    if (inval == 0)
                        continue;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl0);
                    vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt, vl0);
                }

                act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
                oc += (int)vl0;
                continue;
            }

            size_t vl1 = __riscv_vsetvl_e16m4(rem1);
            int oc1 = oc + (int)vl0;
            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[oc], vl0);
            vint32m8_t vacc1 = __riscv_vle32_v_i32m8(&bias_i32[oc1], vl1);
            int col_idx = 0;

            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                const int16_t *wt00 = &weight_buffer[(col_idx + 0) * outC + oc];
                const int16_t *wt10 = &weight_buffer[(col_idx + 1) * outC + oc];
                const int16_t *wt20 = &weight_buffer[(col_idx + 2) * outC + oc];
                const int16_t *wt30 = &weight_buffer[(col_idx + 3) * outC + oc];
                const int16_t *wt40 = &weight_buffer[(col_idx + 4) * outC + oc];
                const int16_t *wt50 = &weight_buffer[(col_idx + 5) * outC + oc];
                const int16_t *wt60 = &weight_buffer[(col_idx + 6) * outC + oc];
                const int16_t *wt70 = &weight_buffer[(col_idx + 7) * outC + oc];
                const int16_t *wt01 = wt00 + vl0;
                const int16_t *wt11 = wt10 + vl0;
                const int16_t *wt21 = wt20 + vl0;
                const int16_t *wt31 = wt30 + vl0;
                const int16_t *wt41 = wt40 + vl0;
                const int16_t *wt51 = wt50 + vl0;
                const int16_t *wt61 = wt60 + vl0;
                const int16_t *wt71 = wt70 + vl0;

                if (inval0 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt00, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt01, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, v1, vl1); }
                if (inval1 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt10, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt11, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, v1, vl1); }
                if (inval2 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt20, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt21, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, v1, vl1); }
                if (inval3 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt30, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt31, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, v1, vl1); }
                if (inval4 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt40, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt41, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval4, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval4, v1, vl1); }
                if (inval5 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt50, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt51, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval5, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval5, v1, vl1); }
                if (inval6 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt60, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt61, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval6, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval6, v1, vl1); }
                if (inval7 != 0) { vint16m4_t v0 = __riscv_vle16_v_i16m4(wt70, vl0); vint16m4_t v1 = __riscv_vle16_v_i16m4(wt71, vl1); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval7, v0, vl0); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval7, v1, vl1); }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                const int16_t *wt0 = &weight_buffer[col_idx * outC + oc];
                const int16_t *wt1 = wt0 + vl0;
                vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl0);
                vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl1);
                vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, vwt0, vl0);
                vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, vwt1, vl1);
            }

            act_kernel(vacc0, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl0);
            act_kernel(vacc1, &M[oc1], &Z[oc1], &output_i8[pos * outC + oc1], vl1);
            oc += (int)(vl0 + vl1);
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc4_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc4_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[4] = {0};
            size_t vls[4] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 4) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[bases[0]], vls[0]);
            vint32m8_t vacc1, vacc2, vacc3;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m8(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m8(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m8(&bias_i32[bases[3]], vls[3]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval7, v, vls[3]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, v, vls[3]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_im2col_unroll8_acc8_m8(NNModule *layer, void *input, void *output)
{
    void *im2col_input = input_im2col_create_nhwc_1d(layer, input);
    if (!im2col_input) {
        printf("Error: im2col_input is NULL in conv1d_i8_vpu_im2col_unroll8_acc8_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int cols = layer->params.conv.filterSize * layer->inputShape.C;

    const int8_t *input_i8 = (const int8_t *)im2col_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv;
    const float *M = (const float *)layer->params.conv.M;
    const int32_t *Z = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t act_kernel =
        select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        const int8_t *input_row = &input_i8[pos * cols];
        int oc = 0;
        while (oc < outC) {
            int bases[8] = {0};
            size_t vls[8] = {0};
            int blocks = 0;
            int cursor = oc;
            while (cursor < outC && blocks < 8) {
                bases[blocks] = cursor;
                vls[blocks] = __riscv_vsetvl_e16m4(outC - cursor);
                cursor += (int)vls[blocks];
                ++blocks;
            }

            vint32m8_t vacc0 = __riscv_vle32_v_i32m8(&bias_i32[bases[0]], vls[0]);
            vint32m8_t vacc1, vacc2, vacc3, vacc4, vacc5, vacc6, vacc7;
            if (blocks > 1) vacc1 = __riscv_vle32_v_i32m8(&bias_i32[bases[1]], vls[1]);
            if (blocks > 2) vacc2 = __riscv_vle32_v_i32m8(&bias_i32[bases[2]], vls[2]);
            if (blocks > 3) vacc3 = __riscv_vle32_v_i32m8(&bias_i32[bases[3]], vls[3]);
            if (blocks > 4) vacc4 = __riscv_vle32_v_i32m8(&bias_i32[bases[4]], vls[4]);
            if (blocks > 5) vacc5 = __riscv_vle32_v_i32m8(&bias_i32[bases[5]], vls[5]);
            if (blocks > 6) vacc6 = __riscv_vle32_v_i32m8(&bias_i32[bases[6]], vls[6]);
            if (blocks > 7) vacc7 = __riscv_vle32_v_i32m8(&bias_i32[bases[7]], vls[7]);

            int col_idx = 0;
            for (; col_idx + 7 < cols; col_idx += 8) {
                int16_t inval0 = (int16_t)input_row[col_idx + 0];
                int16_t inval1 = (int16_t)input_row[col_idx + 1];
                int16_t inval2 = (int16_t)input_row[col_idx + 2];
                int16_t inval3 = (int16_t)input_row[col_idx + 3];
                int16_t inval4 = (int16_t)input_row[col_idx + 4];
                int16_t inval5 = (int16_t)input_row[col_idx + 5];
                int16_t inval6 = (int16_t)input_row[col_idx + 6];
                int16_t inval7 = (int16_t)input_row[col_idx + 7];
                if (inval0 == 0 && inval1 == 0 && inval2 == 0 && inval3 == 0 &&
                    inval4 == 0 && inval5 == 0 && inval6 == 0 && inval7 == 0)
                    continue;
                if (blocks > 0) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[0]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[0]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[0]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[0]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[0]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[0]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[0]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[0]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval0, v, vls[0]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval1, v, vls[0]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval2, v, vls[0]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval3, v, vls[0]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval4, v, vls[0]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval5, v, vls[0]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval6, v, vls[0]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval7, v, vls[0]); }
                }
                if (blocks > 1) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[1]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[1]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[1]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[1]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[1]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[1]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[1]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[1]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval0, v, vls[1]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval1, v, vls[1]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval2, v, vls[1]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval3, v, vls[1]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval4, v, vls[1]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval5, v, vls[1]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval6, v, vls[1]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval7, v, vls[1]); }
                }
                if (blocks > 2) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[2]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[2]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[2]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[2]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[2]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[2]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[2]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[2]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval0, v, vls[2]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval1, v, vls[2]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval2, v, vls[2]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval3, v, vls[2]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval4, v, vls[2]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval5, v, vls[2]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval6, v, vls[2]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval7, v, vls[2]); }
                }
                if (blocks > 3) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[3]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[3]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[3]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[3]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[3]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[3]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[3]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[3]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval0, v, vls[3]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval1, v, vls[3]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval2, v, vls[3]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval3, v, vls[3]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval4, v, vls[3]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval5, v, vls[3]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval6, v, vls[3]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval7, v, vls[3]); }
                }
                if (blocks > 4) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[4]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[4]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[4]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[4]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[4]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[4]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[4]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[4]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval0, v, vls[4]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval1, v, vls[4]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval2, v, vls[4]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval3, v, vls[4]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval4, v, vls[4]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval5, v, vls[4]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval6, v, vls[4]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval7, v, vls[4]); }
                }
                if (blocks > 5) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[5]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[5]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[5]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[5]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[5]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[5]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[5]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[5]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval0, v, vls[5]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval1, v, vls[5]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval2, v, vls[5]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval3, v, vls[5]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval4, v, vls[5]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval5, v, vls[5]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval6, v, vls[5]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval7, v, vls[5]); }
                }
                if (blocks > 6) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[6]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[6]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[6]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[6]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[6]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[6]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[6]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[6]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval0, v, vls[6]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval1, v, vls[6]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval2, v, vls[6]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval3, v, vls[6]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval4, v, vls[6]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval5, v, vls[6]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval6, v, vls[6]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval7, v, vls[6]); }
                }
                if (blocks > 7) {
                    const int16_t *wt0 = &weight_buffer[(col_idx + 0) * outC + bases[7]];
                    const int16_t *wt1 = &weight_buffer[(col_idx + 1) * outC + bases[7]];
                    const int16_t *wt2 = &weight_buffer[(col_idx + 2) * outC + bases[7]];
                    const int16_t *wt3 = &weight_buffer[(col_idx + 3) * outC + bases[7]];
                    const int16_t *wt4 = &weight_buffer[(col_idx + 4) * outC + bases[7]];
                    const int16_t *wt5 = &weight_buffer[(col_idx + 5) * outC + bases[7]];
                    const int16_t *wt6 = &weight_buffer[(col_idx + 6) * outC + bases[7]];
                    const int16_t *wt7 = &weight_buffer[(col_idx + 7) * outC + bases[7]];
                    if (inval0 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt0, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval0, v, vls[7]); }
                    if (inval1 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt1, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval1, v, vls[7]); }
                    if (inval2 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt2, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval2, v, vls[7]); }
                    if (inval3 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt3, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval3, v, vls[7]); }
                    if (inval4 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt4, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval4, v, vls[7]); }
                    if (inval5 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt5, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval5, v, vls[7]); }
                    if (inval6 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt6, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval6, v, vls[7]); }
                    if (inval7 != 0) { vint16m4_t v = __riscv_vle16_v_i16m4(wt7, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval7, v, vls[7]); }
                }
            }

            for (; col_idx < cols; ++col_idx) {
                int16_t inval = (int16_t)input_row[col_idx];
                if (inval == 0)
                    continue;
                if (blocks > 0) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[0]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[0]); vacc0 = __riscv_vwmacc_vx_i32m8(vacc0, inval, v, vls[0]); }
                if (blocks > 1) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[1]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[1]); vacc1 = __riscv_vwmacc_vx_i32m8(vacc1, inval, v, vls[1]); }
                if (blocks > 2) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[2]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[2]); vacc2 = __riscv_vwmacc_vx_i32m8(vacc2, inval, v, vls[2]); }
                if (blocks > 3) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[3]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[3]); vacc3 = __riscv_vwmacc_vx_i32m8(vacc3, inval, v, vls[3]); }
                if (blocks > 4) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[4]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[4]); vacc4 = __riscv_vwmacc_vx_i32m8(vacc4, inval, v, vls[4]); }
                if (blocks > 5) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[5]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[5]); vacc5 = __riscv_vwmacc_vx_i32m8(vacc5, inval, v, vls[5]); }
                if (blocks > 6) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[6]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[6]); vacc6 = __riscv_vwmacc_vx_i32m8(vacc6, inval, v, vls[6]); }
                if (blocks > 7) { const int16_t *wt = &weight_buffer[col_idx * outC + bases[7]]; vint16m4_t v = __riscv_vle16_v_i16m4(wt, vls[7]); vacc7 = __riscv_vwmacc_vx_i32m8(vacc7, inval, v, vls[7]); }
            }

            act_kernel(vacc0, &M[bases[0]], &Z[bases[0]], &output_i8[pos * outC + bases[0]], vls[0]);
            if (blocks > 1) act_kernel(vacc1, &M[bases[1]], &Z[bases[1]], &output_i8[pos * outC + bases[1]], vls[1]);
            if (blocks > 2) act_kernel(vacc2, &M[bases[2]], &Z[bases[2]], &output_i8[pos * outC + bases[2]], vls[2]);
            if (blocks > 3) act_kernel(vacc3, &M[bases[3]], &Z[bases[3]], &output_i8[pos * outC + bases[3]], vls[3]);
            if (blocks > 4) act_kernel(vacc4, &M[bases[4]], &Z[bases[4]], &output_i8[pos * outC + bases[4]], vls[4]);
            if (blocks > 5) act_kernel(vacc5, &M[bases[5]], &Z[bases[5]], &output_i8[pos * outC + bases[5]], vls[5]);
            if (blocks > 6) act_kernel(vacc6, &M[bases[6]], &Z[bases[6]], &output_i8[pos * outC + bases[6]], vls[6]);
            if (blocks > 7) act_kernel(vacc7, &M[bases[7]], &Z[bases[7]], &output_i8[pos * outC + bases[7]], vls[7]);
            oc = cursor;
        }
    }

    safe_free(im2col_input);
}

void conv1d_i8_vpu_chaining2_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 1 < inC; ic += 2) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];


                    if (inval0 != 0 && inval1 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                    } else {
                        if (inval0 != 0) {
                            vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        }
                        if (inval1 != 0) {
                            vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                        }
                    }
                }

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_chaining4_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);
            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 3 < inC; ic += 4) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];

                    if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                        vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                        vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                    } else {
                        if (inval0 != 0) {
                            vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        }
                        if (inval1 != 0) {
                            vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                        }
                        if (inval2 != 0) {
                            vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                        }
                        if (inval3 != 0) {
                            vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                        }
                    }
                }

                for (; ic < inC; ++ic) {
                    int8_t inval = in_ptr[ic];
                    if (inval == 0) {
                        continue;
                    }
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }
    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

void conv1d_i8_vpu_chaining8_m8(NNModule *layer, void *input, void *output){
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu_m8.\n");
        exit(EXIT_FAILURE);
    }

    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inC  = layer->inputShape.C;
    int filterSize = layer->params.conv.filterSize;
    int stride  = layer->params.conv.stride;

    const int8_t  *input_i8   = (const int8_t *)padded_input;
    int8_t        *output_i8  = (int8_t *)output;
    const int32_t *bias_i32   = (const int32_t *)layer->params.conv.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv.weights_rvv; // (K*inC, outC) 佈局
    const float   *M          = (const float *)layer->params.conv.M;
    const int32_t *Z          = (const int32_t *)layer->params.conv.zps;
    requantize_store_chunk_i8_asym_per_channel_kernel_m8_t rq_activation_kernel = select_requantize_store_chunk_i8_asym_per_channel_kernel_m8(layer->activation);

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&bias_i32[oc], vl);

            for (int k = 0; k < filterSize; ++k) {
                const int8_t *in_ptr = &input_i8[(pos * stride + k) * inC];
                int base_col = k * inC;
                int ic = 0;

                for (; ic + 7 < inC; ic += 8) {
                    int16_t inval0 = (int16_t)in_ptr[ic + 0];
                    int16_t inval1 = (int16_t)in_ptr[ic + 1];
                    int16_t inval2 = (int16_t)in_ptr[ic + 2];
                    int16_t inval3 = (int16_t)in_ptr[ic + 3];
                    int16_t inval4 = (int16_t)in_ptr[ic + 4];
                    int16_t inval5 = (int16_t)in_ptr[ic + 5];
                    int16_t inval6 = (int16_t)in_ptr[ic + 6];
                    int16_t inval7 = (int16_t)in_ptr[ic + 7];

                    int col_idx0 = base_col + ic + 0;
                    int col_idx1 = base_col + ic + 1;
                    int col_idx2 = base_col + ic + 2;
                    int col_idx3 = base_col + ic + 3;
                    int col_idx4 = base_col + ic + 4;
                    int col_idx5 = base_col + ic + 5;
                    int col_idx6 = base_col + ic + 6;
                    int col_idx7 = base_col + ic + 7;

                    const int16_t *wt0 = &weight_buffer[col_idx0 * outC + oc];
                    const int16_t *wt1 = &weight_buffer[col_idx1 * outC + oc];
                    const int16_t *wt2 = &weight_buffer[col_idx2 * outC + oc];
                    const int16_t *wt3 = &weight_buffer[col_idx3 * outC + oc];
                    const int16_t *wt4 = &weight_buffer[col_idx4 * outC + oc];
                    const int16_t *wt5 = &weight_buffer[col_idx5 * outC + oc];
                    const int16_t *wt6 = &weight_buffer[col_idx6 * outC + oc];
                    const int16_t *wt7 = &weight_buffer[col_idx7 * outC + oc];

                    if (inval0 != 0 && inval1 != 0 && inval2 != 0 && inval3 != 0 &&
                        inval4 != 0 && inval5 != 0 && inval6 != 0 && inval7 != 0) {
                        vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                        vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                        vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                        vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
   
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);

                        vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt4, vl);
                        vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt5, vl);
                        vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt6, vl);
                        vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt7, vl);
                        
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt4, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt5, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt6, vl);
                        vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt7, vl);
                    } else {
                        if (inval0 != 0) {
                            vint16m4_t vwt0 = __riscv_vle16_v_i16m4(wt0, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval0, vwt0, vl);
                        }
                        if (inval1 != 0) {
                            vint16m4_t vwt1 = __riscv_vle16_v_i16m4(wt1, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval1, vwt1, vl);
                        }
                        if (inval2 != 0) {
                            vint16m4_t vwt2 = __riscv_vle16_v_i16m4(wt2, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval2, vwt2, vl);
                        }
                        if (inval3 != 0) {
                            vint16m4_t vwt3 = __riscv_vle16_v_i16m4(wt3, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval3, vwt3, vl);
                        }
                        if (inval4 != 0) {
                            vint16m4_t vwt4 = __riscv_vle16_v_i16m4(wt4, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval4, vwt4, vl);
                        }
                        if (inval5 != 0) {
                            vint16m4_t vwt5 = __riscv_vle16_v_i16m4(wt5, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval5, vwt5, vl);
                        }
                        if (inval6 != 0) {
                            vint16m4_t vwt6 = __riscv_vle16_v_i16m4(wt6, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval6, vwt6, vl);
                        }
                        if (inval7 != 0) {
                            vint16m4_t vwt7 = __riscv_vle16_v_i16m4(wt7, vl);
                            vacc = __riscv_vwmacc_vx_i32m8(vacc, inval7, vwt7, vl);
                        }
                    }
                }

                for (; ic < inC; ++ic) {
                    int16_t inval = (int16_t)in_ptr[ic];
                    if (inval == 0) continue;
                    int col_idx = base_col + ic;
                    const int16_t *wt = &weight_buffer[col_idx * outC + oc];
                    vint16m4_t vwt = __riscv_vle16_v_i16m4(wt, vl);
                    vacc = __riscv_vwmacc_vx_i32m8(vacc, inval, vwt, vl);
                }
            }

            rq_activation_kernel(vacc, &M[oc], &Z[oc], &output_i8[pos * outC + oc], vl);
            oc += (int)vl;
        }
    }

    if (layer->params.conv.padding > 0) {
        safe_free(padded_input);
    }
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
