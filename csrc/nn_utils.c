#include "nn_utils.h"
/* Utility functions for 1D CNN on embedded RISC-V System */

#define ALIGNMENT 64 // 64-byte alignment for cache line
#define malloc(size)   DO_NOT_USE_MALLOC_USE_SAFE_MALLOC
#define calloc(n, s)   DO_NOT_USE_CALLOC_USE_SAFE_CALLOC

#define DEFINE_ACTIVATE(TYPE, SUFFIX) \
TYPE activate_##SUFFIX(TYPE x, ActivationType type) { \
    switch (type) { \
        case RELU: return x > 0 ? x : (TYPE)0; \
        case LEAKY_RELU: return x > 0 ? x : (TYPE)(0.01 * x); \
        case SIGMOID: { \
            float fx = (float)x; \
            return (TYPE)(1.0f / (1.0f + expf(-fx))); \
        } \
        case TANH: { \
            float fx = (float)x; \
            return (TYPE)tanhf(fx); \
        } \
        case NONE: default: return x; \
    } \
}

DEFINE_ACTIVATE(float, f32);
DEFINE_ACTIVATE(int32_t, i32);
DEFINE_ACTIVATE(int8_t, i8);

void softmax_f32(const float *input, float *output, int len) {
    float max_val = -INFINITY;
    for (int i = 0; i < len; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    double inv_sum = 1.0 / sum;
    for (int i = 0; i < len; i++) {
        output[i] = (float)(output[i] * inv_sum);
    }
}


size_t sizeof_dtype(elem_type t) {
    switch (t) {
        case ELEM_INT8:    return sizeof(int8_t);
        case ELEM_INT16:   return sizeof(int16_t);
        case ELEM_FLOAT16: return 2;
        case ELEM_FLOAT32: return sizeof(float);
        default: return 1;
    }
}

void *safe_malloc(size_t size) {
    size_t aligned_size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    void *ptr = aligned_alloc(ALIGNMENT, aligned_size);
    if (!ptr) {
        fprintf(stderr, "aligned_alloc failed for size %zu (alignment %d)\n", size, ALIGNMENT);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t num, size_t size) {
    size_t total = num * size;
    size_t aligned_size = (total + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    void *ptr = aligned_alloc(ALIGNMENT, aligned_size);
    if (!ptr) {
        fprintf(stderr, "aligned_alloc failed for calloc size %zu (alignment %d)\n", aligned_size, ALIGNMENT);
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, aligned_size);
    return ptr;
}

void safe_free(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

void *padded_input_create_nchw(NNModule *layer, const void *input) {
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int padding = layer->params.conv.padding;
    if (padding == 0)
        return (void *)input;
    elem_type dtype = layer->dtype;

    // 1d padding
    int paddedW = inW + 2 * padding;
    size_t total_size = sizeof_dtype(dtype) * inC * paddedW;
    void *padded_input = safe_malloc(total_size);

    switch (dtype) {
        case ELEM_FLOAT32: {
            float *dst = (float *)padded_input;
            float *src = (float *)input;

            for (int ic = 0; ic < inC; ic++) {
                float *dst_c = &dst[ic * paddedW];
                float *src_c = &src[ic * inW];

                    memset(dst_c, 0, padding * sizeof(float));               // 左 padding
                    memcpy(dst_c + padding, src_c, inW * sizeof(float));     // 原始輸入
                    memset(dst_c + padding + inW, 0, padding * sizeof(float)); // 右 padding
            }
            break;
        }

        case ELEM_INT8: {
            int8_t *dst = (int8_t *)padded_input;
            int8_t *src = (int8_t *)input;

            for (int ic = 0; ic < inC; ic++) {
                int8_t *dst_c = &dst[ic * paddedW];
                int8_t *src_c = &src[ic * inW];

                memset(dst_c, 0, padding * sizeof(int8_t));             // 左 padding
                memcpy(dst_c + padding, src_c, inW * sizeof(int8_t));   // 原始輸入
                memset(dst_c + padding + inW, 0, padding * sizeof(int8_t)); // 右 padding
            }
            break;
        }

        default: {
            printf("Unsupported dtype in padding\n");
            exit(EXIT_FAILURE);
        }
    }
    return padded_input;
}

void *padded_input_create_nhwc(NNModule *layer, const void *input) {
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int padding = layer->params.conv.padding;
    if (padding == 0)
        return (void *)input;
    elem_type dtype = layer->dtype;
    int paddedW = inW + 2 * padding;
    void *padded_input = safe_malloc(sizeof_dtype(dtype) * paddedW * inC);

    switch (dtype) {
        case ELEM_FLOAT32: {
            float *dst = (float *)padded_input;
            const float *src = (const float *)input;

            // NHWC: 以「位置」為主，每個位置有 inC 個 channel
            for (int w = 0; w < paddedW; w++) {
                float *dst_pos = &dst[w * inC];
                if (w < padding || w >= (inW + padding)) {
                    memset(dst_pos, 0, inC * sizeof(float));
                } else {
                    const float *src_pos = &src[(w - padding) * inC];
                    memcpy(dst_pos, src_pos, inC * sizeof(float));
                }
            }
            break;
        }

        case ELEM_INT8: {
            int8_t *dst = (int8_t *)padded_input;
            const int8_t *src = (const int8_t *)input;

            for (int w = 0; w < paddedW; w++) {
                int8_t *dst_pos = &dst[w * inC];
                if (w < padding || w >= (inW + padding)) {
                    memset(dst_pos, 0, inC * sizeof(int8_t));
                } else {
                    const int8_t *src_pos = &src[(w - padding) * inC];
                    memcpy(dst_pos, src_pos, inC * sizeof(int8_t));
                }
            }
            break;
        }

        default:
            printf("Unsupported dtype in padding\n");
            exit(EXIT_FAILURE);
    }

    return padded_input;
}

void *padded_input_create_nchw_2d(NNModule *layer, const void *input) {
    int inH = layer->inputShape.H;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int padding = layer->params.conv2d.padding;
    elem_type dtype = layer->dtype;
    int paddedH = inH + 2 * padding;
    int paddedW = inW + 2 * padding;

    if (padding == 0)
        return (void *)input;

    size_t total_size = sizeof_dtype(dtype) * inC * paddedH * paddedW;
    void *padded_input = safe_malloc(total_size);

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *dst = (float *)padded_input;
        const float *src = (const float *)input;
        for (int c = 0; c < inC; ++c) {
            float *dst_c = &dst[c * paddedH * paddedW];
            const float *src_c = &src[c * inH * inW];
            for (int h = 0; h < paddedH; ++h) {
                float *dst_row = &dst_c[h * paddedW];
                if (h < padding || h >= (inH + padding)) {
                    memset(dst_row, 0, paddedW * sizeof(float));
                } else {
                    const float *src_row = &src_c[(h - padding) * inW];
                    memset(dst_row, 0, padding * sizeof(float));
                    memcpy(dst_row + padding, src_row, inW * sizeof(float));
                    memset(dst_row + padding + inW, 0, padding * sizeof(float));
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *dst = (int8_t *)padded_input;
        const int8_t *src = (const int8_t *)input;
        for (int c = 0; c < inC; ++c) {
            int8_t *dst_c = &dst[c * paddedH * paddedW];
            const int8_t *src_c = &src[c * inH * inW];
            for (int h = 0; h < paddedH; ++h) {
                int8_t *dst_row = &dst_c[h * paddedW];
                if (h < padding || h >= (inH + padding)) {
                    memset(dst_row, 0, paddedW * sizeof(int8_t));
                } else {
                    const int8_t *src_row = &src_c[(h - padding) * inW];
                    memset(dst_row, 0, padding * sizeof(int8_t));
                    memcpy(dst_row + padding, src_row, inW * sizeof(int8_t));
                    memset(dst_row + padding + inW, 0, padding * sizeof(int8_t));
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in 2d NCHW padding\n");
        exit(EXIT_FAILURE);
    }

    return padded_input;
}

void *padded_input_create_nhwc_2d(NNModule *layer, const void *input) {
    int inH = layer->inputShape.H;
    int inW = layer->inputShape.W;
    int inC = layer->inputShape.C;
    int padding = 0;
    if (layer->type == CONV2D) {
        padding = layer->params.conv2d.padding;
    } else if (layer->type == POOL2D) {
        padding = layer->params.pool2d.padding;
    }
    elem_type dtype = layer->dtype;
    int paddedH = inH + 2 * padding;
    int paddedW = inW + 2 * padding;

    if (padding == 0)
        return (void *)input;

    size_t total_size = sizeof_dtype(dtype) * paddedH * paddedW * inC;
    void *padded_input = safe_malloc(total_size);

    switch (dtype) {
    case ELEM_FLOAT32: {
        float *dst = (float *)padded_input;
        const float *src = (const float *)input;
        for (int h = 0; h < paddedH; ++h) {
            for (int w = 0; w < paddedW; ++w) {
                float *dst_pos = &dst[(h * paddedW + w) * inC];
                if (h < padding || h >= (inH + padding) || w < padding || w >= (inW + padding)) {
                    memset(dst_pos, 0, inC * sizeof(float));
                } else {
                    const float *src_pos = &src[((h - padding) * inW + (w - padding)) * inC];
                    memcpy(dst_pos, src_pos, inC * sizeof(float));
                }
            }
        }
        break;
    }
    case ELEM_INT8: {
        int8_t *dst = (int8_t *)padded_input;
        const int8_t *src = (const int8_t *)input;
        for (int h = 0; h < paddedH; ++h) {
            for (int w = 0; w < paddedW; ++w) {
                int8_t *dst_pos = &dst[(h * paddedW + w) * inC];
                if (h < padding || h >= (inH + padding) || w < padding || w >= (inW + padding)) {
                    memset(dst_pos, 0, inC * sizeof(int8_t));
                } else {
                    const int8_t *src_pos = &src[((h - padding) * inW + (w - padding)) * inC];
                    memcpy(dst_pos, src_pos, inC * sizeof(int8_t));
                }
            }
        }
        break;
    }
    default:
        printf("Unsupported dtype in 2d NHWC padding\n");
        exit(EXIT_FAILURE);
    }

    return padded_input;
}


int8_t clip_i32_i8(int32_t x, int8_t min_val, int8_t max_val) {
    // asymmetric clipping
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// bitwidth, scale sharing, zero point type

int8_t requantize_int8_symmetric(const int32_t input, float scale) {
    /*
        int32 accumulator (MAC 結果)
        這裡對應 (s_in * s_w / s_out)
        輸出 z_out
    */
    int32_t q = (int32_t)roundf(input * scale);
    return clip_i32_i8(q, INT8_MIN+1, INT8_MAX); // -128 127
}

int8_t requantize_int8_asymmetric(const int32_t input, float scale, int32_t zero_point) {
    /*
        int32 accumulator (MAC 結果)
        這裡對應 (s_in * s_w / s_out)
        輸出 z_out
    */
    int32_t q = (int32_t)roundf(input * scale) + zero_point;
    return clip_i32_i8(q, INT8_MIN+1, INT8_MAX); // -128 127
}

void free_pingpong_buffer() {
    safe_free(buffer1);
    safe_free(buffer2);
}

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.All rights reserved.
Author : Stanley Lee
*/
