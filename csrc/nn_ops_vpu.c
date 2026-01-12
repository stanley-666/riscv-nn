#include "nn_layer.h"
#include "nn_utils.h"

#include "nn_ops_vpu.h"
#include <time.h>

#define VLEN (__riscv_vlenb() * 8) // Get CSR VLEN Bits

void requantize_activate_store_rvv(const int32_t *src,
                                                 const int32_t *bias,
                                                 const float *scale,
                                                 const int32_t *zp,
                                                 int8_t *dst,
                                                 int len,
                                                 ActivationType act)
{
    if (act == NONE) { // only store
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&src[i], vl);
            vint32m8_t vbias = __riscv_vle32_v_i32m8(&bias[i], vl);
            vacc = __riscv_vadd_vv_i32m8(vacc, vbias, vl);

            vfloat32m8_t vacc_f = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
            vfloat32m8_t vscale = __riscv_vle32_v_f32m8(&scale[i], vl);
            vacc_f = __riscv_vfmul_vv_f32m8(vacc_f, vscale, vl);

            vint32m8_t vround = __riscv_vfcvt_x_f_v_i32m8(vacc_f, vl);
            vint32m8_t vzp = __riscv_vle32_v_i32m8(&zp[i], vl);
            vround = __riscv_vadd_vv_i32m8(vround, vzp, vl);

            vround = __riscv_vmax_vx_i32m8(vround, INT8_MIN, vl);
            vround = __riscv_vmin_vx_i32m8(vround, INT8_MAX, vl);

            vint16m4_t vq16 = __riscv_vncvt_x_x_w_i16m4(vround, vl);
            vint8m2_t vq8 = __riscv_vncvt_x_x_w_i8m2(vq16, vl);
            __riscv_vse8_v_i8m2(&dst[i], vq8, vl);
            i += vl;
        }
        return;
    }
    
    else if (act == RELU) {
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vint32m8_t vacc = __riscv_vle32_v_i32m8(&src[i], vl);
            vint32m8_t vbias = __riscv_vle32_v_i32m8(&bias[i], vl);
            vacc = __riscv_vadd_vv_i32m8(vacc, vbias, vl);

            vfloat32m8_t vacc_f = __riscv_vfcvt_f_x_v_f32m8(vacc, vl);
            vfloat32m8_t vscale = __riscv_vle32_v_f32m8(&scale[i], vl);
            vacc_f = __riscv_vfmul_vv_f32m8(vacc_f, vscale, vl);

            vint32m8_t vround = __riscv_vfcvt_x_f_v_i32m8(vacc_f, vl);
            vint32m8_t vzp = __riscv_vle32_v_i32m8(&zp[i], vl);
            vround = __riscv_vadd_vv_i32m8(vround, vzp, vl);

            vround = __riscv_vmax_vx_i32m8(vround, INT8_MIN, vl);
            vround = __riscv_vmin_vx_i32m8(vround, INT8_MAX, vl);

            vint16m4_t vq16 = __riscv_vncvt_x_x_w_i16m4(vround, vl);
            vint8m2_t vq8 = __riscv_vncvt_x_x_w_i8m2(vq16, vl);

            vint8m2_t vzero = __riscv_vmv_v_x_i8m2(0, vl);
            vq8 = __riscv_vmax_vv_i8m2(vq8, vzero, vl);
  
            __riscv_vse8_v_i8m2(&dst[i], vq8, vl);
            i += vl;
        }
        return;
    }

    /*
    else if(act = LEAKY_RELU) {
        // to be implemented
    }
    */

    for (int i = 0; i < len; ++i) {
        int32_t acc = src[i] + bias[i];
        int8_t rq = requantize_int8(acc, scale[i], zp[i]);
        dst[i] = activate_i8(rq, act);
    }
}

void activate_store_rvv_f32(float *dst,
                            const float *src,
                            int len,
                            ActivationType act)
{

    if (act == RELU) {
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t vdata = __riscv_vle32_v_f32m8(&src[i], vl);
            vfloat32m8_t vzero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
            vdata = __riscv_vfmax_vv_f32m8(vdata, vzero, vl);
            __riscv_vse32_v_f32m8(&dst[i], vdata, vl);
            i += vl;
        }
        return;
    }

    else if (act == SOFTMAX) {
        softmax_f32(src, dst, len);
        return;
    }

    else if (act == NONE) { // only store
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t vdata = __riscv_vle32_v_f32m8(&src[i], vl);
            __riscv_vse32_v_f32m8(&dst[i], vdata, vl);
            i += vl;
        }
        return;
    }
    else if (act == LEAKY_RELU) {
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

    for (int i = 0; i < len; ++i)
        dst[i] = activate_f32(src[i], act);
}
/*
RVV implementation of 1D CNN layers

use NHWC format for input and output tensors

*/

void conv1d_i8_vpu(NNModule *layer, void *input, void *output)
{
    // Input: NHWC (N=1, H=1, W=inputLength, C=inC), padded if needed; Output: NHWC.
    void *padded_input = padded_input_create_nhwc(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv1d_i8_vpu.\n");
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
    int32_t *acc_buffer = (int32_t *)layer->params.conv.acc_buffer;

    for (int pos = 0; pos < outW; ++pos) {
        for (int oc = 0; oc < outC; ) {
            size_t vl = __riscv_vsetvl_e16m4(outC - oc);
            vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

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

            __riscv_vse32_v_i32m8(&acc_buffer[oc], vacc, vl);
            oc += vl;
        }

        int8_t *out_row = &output_i8[pos * outC];
        requantize_activate_store_rvv(acc_buffer, bias_i32, M, Z, out_row, outC, layer->activation);
    }

    safe_free(padded_input);
}

void conv1d_fp32_vpu(NNModule *layer, void *input, void *output)
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

    float *acc_buffer = (float *)layer->params.conv.acc_buffer;

    /* Reorder kernel to (K * inC, outC) layout for contiguous vector loads */

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

            __riscv_vse32_v_f32m8(&acc_buffer[oc], vacc, vl);
            oc += vl;
        }
        
        float *out_row = &output_f32[pos * outC];    
        activate_store_rvv_f32(out_row, acc_buffer, outC, layer->activation);
    }

    safe_free(padded_input);
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.
void conv2d_int8_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    int paddedW = inW + 2 * padding;

    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv2d_vpu.\n");
        exit(EXIT_FAILURE);
    }

    const int8_t *input_i8 = (const int8_t *)padded_input;
    int8_t *output_i8 = (int8_t *)output;
    const int32_t *bias_i32 = (const int32_t *)layer->params.conv2d.bias;
    const int16_t *weight_buffer = (const int16_t *)layer->params.conv2d.weights_rvv;
    const float *M = (const float *)layer->params.conv2d.M;
    const int32_t *Z = (const int32_t *)layer->params.conv2d.zps;
    int32_t *acc_buffer = (int32_t *)layer->params.conv2d.acc_buffer;

    if (!weight_buffer) {
        printf("Error: conv2d weights_rvv not initialized.\n");
        exit(EXIT_FAILURE);
    }

    for (int oh = 0, base_h = 0; oh < outH; ++oh, base_h += stride) {
        for (int ow = 0, base_w = 0; ow < outW; ++ow, base_w += stride) {
            const int8_t *base_in = &input_i8[(base_h * paddedW + base_w) * inC];
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e16m4(outC - oc);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int kh = 0; kh < filterSize; ++kh) {
                    const int8_t *row_in = base_in + kh * paddedW * inC;
                    int col_base = kh * filterSize * inC;
                    for (int kw = 0; kw < filterSize; ++kw) {
                        const int8_t *in_ptr = row_in + kw * inC;
                        int col_idx = col_base + kw * inC;
                        const int16_t *wt_base = &weight_buffer[col_idx * outC + oc];
                        for (int ic = 0; ic < inC; ++ic) {
                            int8_t inval = in_ptr[ic];
                            if (inval == 0)
                                continue;
                            vint16m4_t vrow16 = __riscv_vmv_v_x_i16m4(inval, vl);
                            const int16_t *wt = wt_base + ic * outC;
                            vint16m4_t vwt16 = __riscv_vle16_v_i16m4(wt, vl);
                            vacc = __riscv_vwmacc_vv_i32m8(vacc, vrow16, vwt16, vl);
                        }
                    }
                }

                __riscv_vse32_v_i32m8(&acc_buffer[oc], vacc, vl);
                oc += vl;
            }

            int8_t *out_row = &output_i8[(oh * outW + ow) * outC];
            requantize_activate_store_rvv(acc_buffer, bias_i32, M, Z, out_row, outC, layer->activation);
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.
void conv2d_fp32_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1, no dilation.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int outC = layer->outputShape.C;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int filterSize = layer->params.conv2d.filterSize;
    int stride = layer->params.conv2d.stride;
    int padding = layer->params.conv2d.padding;
    int paddedW = inW + 2 * padding;

    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    if (!padded_input) {
        printf("Error: padded_input is NULL in conv2d_vpu.\n");
        exit(EXIT_FAILURE);
    }

    const float *input_f32 = (const float *)padded_input;
    float *output_f32 = (float *)output;
    const float *bias_f32 = (const float *)layer->params.conv2d.bias;
    const float *weight_buffer = (const float *)layer->params.conv2d.weights_rvv;
    float *acc_buffer = (float *)layer->params.conv2d.acc_buffer;

    for (int oh = 0, base_h = 0; oh < outH; ++oh, base_h += stride) {
        for (int ow = 0, base_w = 0; ow < outW; ++ow, base_w += stride) {
            const float *base_in = &input_f32[(base_h * paddedW + base_w) * inC];
            for (int oc = 0; oc < outC; ) {
                size_t vl = __riscv_vsetvl_e32m8(outC - oc);
                vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[oc], vl);

                for (int kh = 0; kh < filterSize; ++kh) {
                    const float *row_in = base_in + kh * paddedW * inC;
                    int col_base = kh * filterSize * inC;
                    for (int kw = 0; kw < filterSize; ++kw) {
                        const float *in_ptr = row_in + kw * inC;
                        int col_idx = col_base + kw * inC;
                        const float *wt_base = &weight_buffer[col_idx * outC + oc];
                        for (int ic = 0; ic < inC; ++ic) {
                            float inval = in_ptr[ic];
                            const float *wt = wt_base + ic * outC;
                            vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt, vl);
                            vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                        }
                    }
                }

                __riscv_vse32_v_f32m8(&acc_buffer[oc], vacc, vl);
                oc += vl;
            }

            float *out_row = &output_f32[(oh * outW + ow) * outC];
            activate_store_rvv_f32(out_row, acc_buffer, outC, layer->activation);
        }
    }

    if (padding > 0)
        safe_free(padded_input);
}

// Input: NHWC with H=1, layout [N=1][W][C]; Output: same layout, W pooled.
void maxpool1d_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;
    elem_type dtype   = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float*)input;
        float *out = (float*)output;

        for (int pos = 0; pos < outW; pos++) {
            int start = pos * stride;
            int end = start + poolSize;
            if (end > inW) end = inW;

            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e32m8(inC - c);
                vfloat32m8_t vmaxv = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);
                for (int w = start; w < end; w++) {
                    const float *ptr = &in[w * inC + c];
                    vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                    vmaxv = __riscv_vfmax_vv_f32m8(vmaxv, vin, vl);
                }
                __riscv_vse32_v_f32m8(&out[pos * inC + c], vmaxv, vl);
                c += (int)vl;
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t*)input;
        int8_t *out = (int8_t*)output;

        for (int pos = 0; pos < outW; pos++) {
            int start = pos * stride;
            int end = start + poolSize;
            if (end > inW) end = inW;

            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e8m8(inC - c);
                vint8m8_t vmaxv = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);
                for (int w = start; w < end; w++) {
                    const int8_t *ptr = &in[w * inC + c];
                    vint8m8_t vin = __riscv_vle8_v_i8m8(ptr, vl);
                    vmaxv = __riscv_vmax_vv_i8m8(vmaxv, vin, vl);
                }
                __riscv_vse8_v_i8m8(&out[pos * inC + c], vmaxv, vl);
                c += (int)vl;
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in maxpool1d_vpu (NHWC)\n");
        exit(EXIT_FAILURE);
    }
}

// Input: NHWC with H=1, layout [N=1][W][C]; Output: same layout, W pooled.
void avgpool1d_vpu(NNModule *layer, void *input, void *output)
{
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool.poolSize;
    int stride   = layer->params.pool.stride;
    elem_type dtype   = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float*)input;
        float *out = (float*)output;

        for (int pos = 0; pos < outW; pos++) {
            int start = pos * stride;
            int end = start + poolSize;
            if (end > inW) end = inW;
            int len = end - start;

            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e32m8(inC - c);
                vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vl);
                for (int w = start; w < end; w++) {
                    const float *ptr = &in[w * inC + c];
                    vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                    vsum = __riscv_vfadd_vv_f32m8(vsum, vin, vl);
                }
                float inv = len > 0 ? 1.0f / (float)len : 0.0f;
                vsum = __riscv_vfmul_vf_f32m8(vsum, inv, vl);
                __riscv_vse32_v_f32m8(&out[pos * inC + c], vsum, vl);
                c += (int)vl;
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t*)input;
        int8_t *out = (int8_t*)output;

        for (int pos = 0; pos < outW; pos++) {
            int start = pos * stride;
            int end = start + poolSize;
            if (end > inW) end = inW;
            int len = end - start;

            for (int c = 0; c < inC; ) {
                size_t vl = __riscv_vsetvl_e8m4(inC - c);
                vint16m8_t vsum = __riscv_vmv_v_x_i16m8(0, vl);
                for (int w = start; w < end; w++) {
                    const int8_t *ptr = &in[w * inC + c];
                    vint8m4_t vin8 = __riscv_vle8_v_i8m4(ptr, vl);
                    vint16m8_t vin16 = __riscv_vsext_vf2_i16m8(vin8, vl);
                    vsum = __riscv_vadd_vv_i16m8(vsum, vin16, vl);
                }
                if (len > 0)
                    vsum = __riscv_vdiv_vx_i16m8(vsum, len, vl);
                vsum = __riscv_vmax_vx_i16m8(vsum, INT8_MIN, vl);
                vsum = __riscv_vmin_vx_i16m8(vsum, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum, vl);
                __riscv_vse8_v_i8m4(&out[pos * inC + c], vout, vl);
                c += (int)vl;
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in avgpool1d_vpu (NHWC)\n");
        exit(EXIT_FAILURE);
    }
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.
void maxpool2d_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;
    elem_type dtype   = layer->dtype;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float *)padded_input;
        float *out = (float *)output;
        for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
            for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
                for (int c = 0; c < inC; ) {
                    size_t vl = __riscv_vsetvl_e32m8(inC - c);
                    vfloat32m8_t vmaxv = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);

                    for (int kh = 0; kh < poolSize; ++kh) {
                        int h = h_start + kh;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            int w = w_start + kw;
                            const float *ptr = &in[(h * paddedW + w) * inC + c];
                            vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                            vmaxv = __riscv_vfmax_vv_f32m8(vmaxv, vin, vl);
                        }
                    }

                    __riscv_vse32_v_f32m8(&out[(oh * outW + ow) * inC + c], vmaxv, vl);
                    c += (int)vl;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t *)padded_input;
        int8_t *out = (int8_t *)output;
        for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
            for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
                for (int c = 0; c < inC; ) {
                    size_t vl = __riscv_vsetvl_e8m8(inC - c);
                    vint8m8_t vmaxv = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);
                    const int8_t *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                    for (int kh = 0; kh < poolSize; ++kh) {
                        const int8_t *row = row_ptr + kh * paddedW * inC;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            const int8_t *ptr = row + kw * inC;
                            vint8m8_t vin = __riscv_vle8_v_i8m8(ptr, vl);
                            vmaxv = __riscv_vmax_vv_i8m8(vmaxv, vin, vl);
                        }
                    }
                    __riscv_vse8_v_i8m8(&out[(oh * outW + ow) * inC + c], vmaxv, vl);
                    c += (int)vl;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in maxpool2d_vpu\n");
        exit(EXIT_FAILURE);
    }

    if (padding > 0)
        safe_free(padded_input);
}

// Input: NHWC (N=1, H=inH, W=inW, C=inC) padded if needed; Output: NHWC.
void avgpool2d_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, square kernel/stride/padding, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    int poolSize = layer->params.pool2d.poolSize;
    int stride   = layer->params.pool2d.stride;
    int padding  = layer->params.pool2d.padding;
    elem_type dtype   = layer->dtype;

    int paddedW = inW + 2 * padding;
    void *padded_input = padded_input_create_nhwc_2d(layer, input);
    int window_elems = poolSize * poolSize;

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float *)padded_input;
        float *out = (float *)output;
        for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
            for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
                for (int c = 0; c < inC; ) {
                    size_t vl = __riscv_vsetvl_e32m8(inC - c);
                    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vl);

                    const float *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                    for (int kh = 0; kh < poolSize; ++kh) {
                        const float *row = row_ptr + kh * paddedW * inC;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            const float *ptr = row + kw * inC;
                            vfloat32m8_t vin = __riscv_vle32_v_f32m8(ptr, vl);
                            vsum = __riscv_vfadd_vv_f32m8(vsum, vin, vl);
                        }
                    }

                    if (window_elems > 0) {
                        float inv = 1.0f / (float)window_elems;
                        vsum = __riscv_vfmul_vf_f32m8(vsum, inv, vl);
                    }

                    __riscv_vse32_v_f32m8(&out[(oh * outW + ow) * inC + c], vsum, vl);
                    c += (int)vl;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t *)padded_input;
        int8_t *out = (int8_t *)output;
        for (int oh = 0, h_start = 0; oh < outH; ++oh, h_start += stride) {
            for (int ow = 0, w_start = 0; ow < outW; ++ow, w_start += stride) {
                for (int c = 0; c < inC; ) {
                    size_t vl = __riscv_vsetvl_e8m4(inC - c);
                    vint16m8_t vsum = __riscv_vmv_v_x_i16m8(0, vl);
                    const int8_t *row_ptr = &in[(h_start * paddedW + w_start) * inC + c];
                    for (int kh = 0; kh < poolSize; ++kh) {
                        const int8_t *row = row_ptr + kh * paddedW * inC;
                        for (int kw = 0; kw < poolSize; ++kw) {
                            const int8_t *ptr = row + kw * inC;
                            vint8m4_t vin8 = __riscv_vle8_v_i8m4(ptr, vl);
                            vint16m8_t vin16 = __riscv_vsext_vf2_i16m8(vin8, vl);
                            vsum = __riscv_vadd_vv_i16m8(vsum, vin16, vl);
                        }
                    }

                    if (window_elems > 0) {
                        vsum = __riscv_vdiv_vx_i16m8(vsum, window_elems, vl);
                    }
                    vsum = __riscv_vmax_vx_i16m8(vsum, INT8_MIN, vl);
                    vsum = __riscv_vmin_vx_i16m8(vsum, INT8_MAX, vl);
                    vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum, vl);
                    __riscv_vse8_v_i8m4(&out[(oh * outW + ow) * inC + c], vout, vl);
                    c += (int)vl;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in avgpool2d_vpu\n");
        exit(EXIT_FAILURE);
    }

    if (padding > 0)
        safe_free(padded_input);
}

void AdaptiveAvgPool2d_vpu(NNModule *layer, void *input, void *output)
{
    // Limitations: NHWC layout, batch=1.
    int outH = layer->outputShape.H;
    int outW = layer->outputShape.W;
    int inH  = layer->inputShape.H;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    elem_type dtype   = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        const float *in = (const float *)input;
        float *out = (float *)output;
        for (int oh = 0; oh < outH; ++oh) {
            int h_start = (oh * inH) / outH;
            int h_end = ((oh + 1) * inH) / outH;
            for (int ow = 0; ow < outW; ++ow) {
                int w_start = (ow * inW) / outW;
                int w_end = ((ow + 1) * inW) / outW;
                for (int c = 0; c < inC; ++c) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += in[(h * inW + w) * inC + c];
                            count++;
                        }
                    }
                    out[(oh * outW + ow) * inC + c] = (count > 0) ? sum / count : 0.0f;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        const int8_t *in = (const int8_t *)input;
        int8_t *out = (int8_t *)output;
        for (int oh = 0; oh < outH; ++oh) {
            int h_start = (oh * inH) / outH;
            int h_end = ((oh + 1) * inH) / outH;
            for (int ow = 0; ow < outW; ++ow) {
                int w_start = (ow * inW) / outW;
                int w_end = ((ow + 1) * inW) / outW;
                for (int c = 0; c < inC; ++c) {
                    int32_t sum = 0;
                    int count = 0;
                    for (int h = h_start; h < h_end; ++h) {
                        for (int w = w_start; w < w_end; ++w) {
                            sum += in[(h * inW + w) * inC + c];
                            count++;
                        }
                    }
                    out[(oh * outW + ow) * inC + c] = (count > 0) ? (int8_t)(sum / count) : 0;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in AdaptiveAvgPool2d_vpu\n");
        exit(EXIT_FAILURE);
    }
}

void fullyconnected_vpu(NNModule *layer, void *input, void *output)
{
    // Input: 1D vector length inDim (W* C), Output: length outW; both contiguous.
    int inDim  = layer->inputShape.W * layer->inputShape.C;
    int outW = layer->outputShape.W; // FC 輸出展平成一維
    ActivationType act = layer->activation;

    // 轉置權重為 (inDim, outW) 佈局，方便以「輸出通道」為向量方向做載入
    switch (layer->dtype) {
        case ELEM_FLOAT32: {
            const float *input_f32 = (const float*)input;
            float *output_f32 = (float*)output;
            const float *weights_f32 = (const float*)layer->params.fc.weights;
            const float *bias_f32 = (const float*)layer->params.fc.bias;

            const float *wt_T = (const float *)layer->params.fc.weights_rvv;
            if (!wt_T) {
                // 後備路徑：若未預先重排，現場轉置一次
                float *tmp = (float *)safe_malloc(inDim * outW * sizeof(float));
                for (int o = 0; o < outW; ++o)
                    for (int i = 0; i < inDim; ++i)
                        tmp[i * outW + o] = weights_f32[o * inDim + i];
                wt_T = tmp;
            }

            for (int o = 0; o < outW; ) {
                size_t vl = __riscv_vsetvl_e32m8(outW - o);
                vfloat32m8_t vacc = __riscv_vle32_v_f32m8(&bias_f32[o], vl);

                for (int i = 0; i < inDim; ++i) {
                    float inval = input_f32[i];
                    if (inval == 0.0f)
                        continue;
                    const float *wt_ptr = &wt_T[i * outW + o];
                    vfloat32m8_t vwt = __riscv_vle32_v_f32m8(wt_ptr, vl);
                    vacc = __riscv_vfmacc_vf_f32m8(vacc, inval, vwt, vl);
                }

                __riscv_vse32_v_f32m8(&output_f32[o], vacc, vl);
                o += vl;
            }

            activate_store_rvv_f32(output_f32, output_f32, outW, act);

            if (wt_T != (const float *)layer->params.fc.weights_rvv)
                safe_free((void *)wt_T);
            break;
        }
        case ELEM_INT8: {
            const int8_t *input_i8 = (const int8_t*)input;
            int8_t *output_i8 = (int8_t*)output;
            const int8_t *weight_i8 = (const int8_t*)layer->params.fc.weights;
            const int32_t *bias_i32 = (const int32_t*)layer->params.fc.bias;
            const float *M = (const float *)layer->params.fc.M;
            const int32_t *Z = (const int32_t *)layer->params.fc.zps; // zero point

            if (act == SOFTMAX)
                printf("Warning: INT8 FC softmax not implemented in RVV path.\n");

            const int8_t *wt_T = (const int8_t *)layer->params.fc.weights_rvv;
            if (!wt_T) {
                int8_t *tmp = (int8_t *)safe_malloc(inDim * outW * sizeof(int8_t));
                for (int o = 0; o < outW; ++o)
                    for (int i = 0; i < inDim; ++i)
                        tmp[i * outW + o] = weight_i8[o * inDim + i];
                wt_T = tmp;
            }

            int32_t *acc_buffer = (int32_t *)layer->params.fc.acc_buffer; // 預先配置的 FC 累加暫存
            if (!acc_buffer)
                acc_buffer = (int32_t *)safe_malloc(outW * sizeof(int32_t));

            for (int o = 0; o < outW; ) {
                size_t vl = __riscv_vsetvl_e8m2(outW - o);
                vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int i = 0; i < inDim; ++i) {
                    int8_t inval = input_i8[i];
                    if (inval == 0)
                        continue;
                    const int8_t *wt_ptr = &wt_T[i * outW + o];
                    vint8m2_t vrow8 = __riscv_vmv_v_x_i8m2(inval, vl);
                    vint8m2_t vwt8  = __riscv_vle8_v_i8m2(wt_ptr, vl);
                    vint16m4_t vrow16 = __riscv_vsext_vf2_i16m4(vrow8, vl);
                    vint16m4_t vwt16  = __riscv_vsext_vf2_i16m4(vwt8, vl);
                    vacc = __riscv_vwmacc_vv_i32m8(vacc, vrow16, vwt16, vl);
                }

                __riscv_vse32_v_i32m8(&acc_buffer[o], vacc, vl);
                o += vl;
            }

            requantize_activate_store_rvv(acc_buffer, bias_i32, M, Z, output_i8, outW, act);

            if (acc_buffer != (int32_t *)layer->params.fc.acc_buffer)
                safe_free(acc_buffer);
            if (wt_T != (const int8_t *)layer->params.fc.weights_rvv)
                safe_free((void *)wt_T);
            break;
        }
        default: 
            printf("Unsupported dtype in fc_forward\n");
            exit(EXIT_FAILURE);
    }
}

void AdaptiveMaxPool1d_vpu(NNModule *layer, void *input, void *output) {
    int outW = layer->outputShape.W;
    int inW  = layer->inputShape.W;
    int inC  = layer->inputShape.C;
    elem_type dtype = layer->dtype;

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *in = (float*)input;
        float *out = (float*)output;

        // 快速路徑：global pooling
        if (outW == 1) {
            for (int c = 0; c < inC; c++) {
                float max_val = -INFINITY;
                for (int idx = 0; idx < inW; idx++) {
                    float v = in[c*inW + idx];
                    if (v > max_val) max_val = v;
                }
                out[c] = max_val;
            }
            break;
        }

        // 一般 adaptive pooling
        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                int start = (pos * inW) / outW;
                int end   = ((pos + 1) * inW) / outW;
                if (end > inW) end = inW;

                float value = -INFINITY;
                for (int idx = start; idx < end; idx++) {
                    float v = in[c*inW + idx];
                    if (v > value) value = v;
                }
                out[c*outW + pos] = value;
            }
        }
        break;
    }

    case ELEM_INT8: {
        const int8_t *in = (const int8_t*)input;
        int8_t *out = (int8_t*)output;

        // 快速路徑：global pooling
        if (outW == 1) {
            for (int c = 0; c < inC; c++) {
                int8_t max_val = INT8_MIN;
                for (int idx = 0; idx < inW; idx++) {
                    int8_t v = in[c*inW + idx];
                    if (v > max_val) max_val = v;
                }
                out[c] = max_val;
            }
            break;
        }

        // 一般 adaptive pooling
        for (int c = 0; c < inC; c++) {
            for (int pos = 0; pos < outW; pos++) {
                int start = (pos * inW) / outW;
                int end   = ((pos + 1) * inW) / outW;
                if (end > inW) end = inW;

                int8_t value = INT8_MIN;
                for (int idx = start; idx < end; idx++) {
                    int8_t v = in[c*inW + idx];
                    if (v > value) value = v;
                }
                out[c*outW + pos] = value;
            }
        }
        break;
    }

    default:
        printf("Unsupported dtype in AdaptiveMaxPool1d_forward\n");
        exit(EXIT_FAILURE);
    }
}

void transpose_vpu(NNModule *layer, void *input, void *output)
{
    // Input/Output: N=1,H=1; layout [C][W] or [W][C] depending on mode.
    int width = layer->inputShape.W;
    int channels = layer->inputShape.C;
    TransposeType mode = layer->params.transpose.mode;

    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *out = (float *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c) {
                const float *in_row = &in[c * width];
                float *out_col = &out[c];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e32m8(width - w);
                    vfloat32m8_t v = __riscv_vle32_v_f32m8(&in_row[w], vl);
                    __riscv_vsse32_v_f32m8(&out_col[w * channels],
                                           (ptrdiff_t)(channels * sizeof(float)),
                                           v, vl);
                    w += (int)vl;
                }
            }
        } else {
            for (int c = 0; c < channels; ++c) {
                const float *in_col = &in[c];
                float *out_row = &out[c * width];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e32m8(width - w);
                    vfloat32m8_t v = __riscv_vlse32_v_f32m8(&in_col[w * channels],
                                                           (ptrdiff_t)(channels * sizeof(float)),
                                                           vl);
                    __riscv_vse32_v_f32m8(&out_row[w], v, vl);
                    w += (int)vl;
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *out = (int8_t *)output;
        if (mode == TRANSPOSE_CW_TO_WC) {
            for (int c = 0; c < channels; ++c) {
                const int8_t *in_row = &in[c * width];
                int8_t *out_col = &out[c];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e8m8(width - w);
                    vint8m8_t v = __riscv_vle8_v_i8m8(&in_row[w], vl);
                    __riscv_vsse8_v_i8m8(&out_col[w * channels],
                                         (ptrdiff_t)channels,
                                         v, vl);
                    w += (int)vl;
                }
            }
        } else {
            for (int c = 0; c < channels; ++c) {
                const int8_t *in_col = &in[c];
                int8_t *out_row = &out[c * width];
                for (int w = 0; w < width; ) {
                    size_t vl = __riscv_vsetvl_e8m8(width - w);
                    vint8m8_t v = __riscv_vlse8_v_i8m8(&in_col[w * channels],
                                                      (ptrdiff_t)channels,
                                                      vl);
                    __riscv_vse8_v_i8m8(&out_row[w], v, vl);
                    w += (int)vl;
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in transpose_vpu\n");
        exit(EXIT_FAILURE);
    }
}

void save_vpu(NNModule *layer, void *input, void *output)
{
    // Input: arbitrary tensor in NHWC/1D (no change), copied into params.save.buffer.
    if (!layer->params.save.buffer || layer->params.save.bytes == 0) {
        printf("Save layer error: buffer or bytes not initialized\n");
        exit(EXIT_FAILURE);
    }

    size_t bytes = layer->params.save.bytes;
    int8_t *src_ptr = (int8_t *)input;
    int8_t *skip_ptr = (int8_t *)layer->params.save.buffer;
    (void)output; // output is intentionally untouched to avoid CPU writes

    for (size_t remaining = bytes; remaining > 0; ) {
        size_t vl = __riscv_vsetvl_e8m8(remaining);
        vint8m8_t v = __riscv_vle8_v_i8m8(src_ptr, vl);
        __riscv_vse8_v_i8m8(skip_ptr, v, vl);
        src_ptr += vl;
        skip_ptr += vl;
        remaining -= vl;
    }
}

void add_vpu(NNModule *layer, void *input, void *output)
{
    // Input: main tensor and skip tensor (same shape), both NHWC/1D; Output: same layout.
    if (!layer->params.add.skip || layer->params.add.bytes == 0) {
        printf("Add layer skip buffer not initialized\n");
        exit(EXIT_FAILURE);
    }
    switch (layer->dtype) {
    case ELEM_FLOAT32: {
        float *in = (float *)input;
        float *skip = (float *)layer->params.add.skip;
        float *out = (float *)output;
        int len = (int)(layer->params.add.bytes / sizeof(float));
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e32m8(len - i);
            vfloat32m8_t v0 = __riscv_vle32_v_f32m8(&in[i], vl);
            vfloat32m8_t v1 = __riscv_vle32_v_f32m8(&skip[i], vl);
            vfloat32m8_t vsum = __riscv_vfadd_vv_f32m8(v0, v1, vl);
            __riscv_vse32_v_f32m8(&out[i], vsum, vl);
            i += (int)vl;
        }
        if (layer->activation != NONE)
            activate_store_rvv_f32(out, out, len, layer->activation);
        break;
    }
    case ELEM_INT8: {
        int8_t *in = (int8_t *)input;
        int8_t *skip = (int8_t *)layer->params.add.skip;
        int8_t *out = (int8_t *)output;
        int len = (int)(layer->params.add.bytes / sizeof(int8_t));
        for (int i = 0; i < len; ) {
            size_t vl = __riscv_vsetvl_e8m4(len - i);
            vint8m4_t v0 = __riscv_vle8_v_i8m4(&in[i], vl);
            vint8m4_t v1 = __riscv_vle8_v_i8m4(&skip[i], vl);
            vint16m8_t v0_16 = __riscv_vsext_vf2_i16m8(v0, vl);
            vint16m8_t v1_16 = __riscv_vsext_vf2_i16m8(v1, vl);
            vint16m8_t vsum16 = __riscv_vadd_vv_i16m8(v0_16, v1_16, vl);

            if (layer->activation == NONE) {
                // clamp to int8 then narrow back
                vsum16 = __riscv_vmax_vx_i16m8(vsum16, INT8_MIN, vl);
                vsum16 = __riscv_vmin_vx_i16m8(vsum16, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum16, vl);
                __riscv_vse8_v_i8m4(&out[i], vout, vl);
            } else {
                // store to temp buffer for scalar activation
                // (activation_i8 is scalar; keep old behavior)
                vsum16 = __riscv_vmax_vx_i16m8(vsum16, INT8_MIN, vl);
                vsum16 = __riscv_vmin_vx_i16m8(vsum16, INT8_MAX, vl);
                vint8m4_t vout = __riscv_vncvt_x_x_w_i8m4(vsum16, vl);
                __riscv_vse8_v_i8m4(&out[i], vout, vl);
                for (size_t t = 0; t < vl; ++t)
                    out[i + t] = activate_i8(out[i + t], layer->activation);
            }
            i += (int)vl;
        }
        break;
    }
    default:
        printf("Unsupported dtype in add_vpu\n");
        exit(EXIT_FAILURE);
    }
}



/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.All rights reserved.
Author : Stanley Lee
*/
