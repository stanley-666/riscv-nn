#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#include "random_embedding_gemmini.h"
#include "weights_q_gemmini.h"

#define CONV1_OUT_SCALE 0.66297567f
#define CONV2_OUT_SCALE 2.63254571f
#define CONV3_OUT_SCALE 0.64812046f
#define FC1_OUT_SCALE 2.76002026f
#define FC2_OUT_SCALE 0.30590293f

#define CONV1_IN_CHANNELS IN_CHANNELS
#define CONV1_OUT_CHANNELS 64
#define CONV1_KERNEL 5
#define CONV1_PADDING 2
#define CONV1_STRIDE 1
#define CONV1_OUT_COL_DIM ((IN_COL_DIM + 2 * CONV1_PADDING - CONV1_KERNEL) / CONV1_STRIDE + 1)

#define CONV2_IN_CHANNELS CONV1_OUT_CHANNELS
#define CONV2_OUT_CHANNELS 128
#define CONV2_KERNEL 5
#define CONV2_PADDING 2
#define CONV2_STRIDE 1
#define CONV2_OUT_COL_DIM ((CONV1_OUT_COL_DIM + 2 * CONV2_PADDING - CONV2_KERNEL) / CONV2_STRIDE + 1)

#define CONV3_IN_CHANNELS CONV2_OUT_CHANNELS
#define CONV3_OUT_CHANNELS 256
#define CONV3_KERNEL 3
#define CONV3_PADDING 1
#define CONV3_STRIDE 1
#define CONV3_OUT_COL_DIM ((CONV2_OUT_COL_DIM + 2 * CONV3_PADDING - CONV3_KERNEL) / CONV3_STRIDE + 1)

#define FC1_IN_FEATURES CONV3_OUT_CHANNELS
#define FC1_OUT_FEATURES 128
#define FC2_IN_FEATURES FC1_OUT_FEATURES
#define FC2_OUT_FEATURES 1

static void im2col_1d(int batch_size, int in_col_dim, int in_channels,
    int kernel_size, int padding, int stride, int out_col_dim,
    const elem_t *input, elem_t *output)
{
    const int k_dim = kernel_size * in_channels;

    for (int b = 0; b < batch_size; b++) {
        for (int ocol = 0; ocol < out_col_dim; ocol++) {
            const int row = b * out_col_dim + ocol;

            for (int k = 0; k < kernel_size; k++) {
                const int in_col = ocol * stride + k - padding;
                for (int ch = 0; ch < in_channels; ch++) {
                    elem_t val = 0;
                    if (in_col >= 0 && in_col < in_col_dim) {
                        const size_t idx = (size_t)(b * in_col_dim + in_col) * in_channels + ch;
                        val = input[idx];
                    }
                    output[row * k_dim + (k * in_channels + ch)] = val;
                }
            }
        }
    }
}

static void flatten_conv_weights_1d(int out_channels, int in_channels, int kernel_size,
    const elem_t *weights, elem_t *weights_mat)
{
    const int k_dim = kernel_size * in_channels;

    for (int och = 0; och < out_channels; och++) {
        for (int k = 0; k < kernel_size; k++) {
            for (int ch = 0; ch < in_channels; ch++) {
                const int k_idx = k * in_channels + ch;
                const size_t w_idx = (((size_t)och * kernel_size + k) * in_channels + ch);
                weights_mat[k_idx * out_channels + och] = weights[w_idx];
            }
        }
    }
}

static void flatten_fc_weights(int out_features, int in_features,
    const elem_t *weights, elem_t *weights_mat)
{
    for (int out = 0; out < out_features; out++) {
        for (int in = 0; in < in_features; in++) {
            const size_t w_idx = (size_t)out * in_features + in;
            weights_mat[in * out_features + out] = weights[w_idx];
        }
    }
}

static int32_t round_to_int(float x) {
    return (int32_t)(x >= 0.0f ? x + 0.5f : x - 0.5f);
}

static int8_t clip_i32_i8(int32_t x) {
    if (x < -127) {
        return -127;
    }
    if (x > 127) {
        return 127;
    }
    return (int8_t)x;
}

static int8_t requantize_int8(int32_t input, float scale, int32_t zero_point) {
    int32_t q = round_to_int((float)input * scale) + zero_point;
    return clip_i32_i8(q);
}

static float sigmoidf(float x) {
    if (x <= -8.0f) {
        return 0.0f;
    }
    if (x >= 8.0f) {
        return 1.0f;
    }

    const float x2 = x * x;
    const float x3 = x2 * x;
    const float x5 = x3 * x2;
    float y = 0.5f + (x * 0.25f) - (x3 * (1.0f / 48.0f)) + (x5 * (1.0f / 480.0f));
    return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif

    gemmini_flush(0);

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Conv1 out cols: %u\n", CONV1_OUT_COL_DIM);
    printf("Conv2 out cols: %u\n", CONV2_OUT_COL_DIM);
    printf("Conv3 out cols: %u\n\n", CONV3_OUT_COL_DIM);

    static elem_t conv1_weights_mat[CONV1_KERNEL * CONV1_IN_CHANNELS][CONV1_OUT_CHANNELS];
    static elem_t conv2_weights_mat[CONV2_KERNEL * CONV2_IN_CHANNELS][CONV2_OUT_CHANNELS];
    static elem_t conv3_weights_mat[CONV3_KERNEL * CONV3_IN_CHANNELS][CONV3_OUT_CHANNELS];
    static elem_t fc1_weights_mat[FC1_IN_FEATURES][FC1_OUT_FEATURES];
    static elem_t fc2_weights_mat[FC2_IN_FEATURES][FC2_OUT_FEATURES];

    static elem_t conv1_im2col[BATCH_SIZE * CONV1_OUT_COL_DIM][CONV1_KERNEL * CONV1_IN_CHANNELS];
    static elem_t conv2_im2col[BATCH_SIZE * CONV2_OUT_COL_DIM][CONV2_KERNEL * CONV2_IN_CHANNELS];
    static elem_t conv3_im2col[BATCH_SIZE * CONV3_OUT_COL_DIM][CONV3_KERNEL * CONV3_IN_CHANNELS];

    static acc_t conv1_acc[BATCH_SIZE * CONV1_OUT_COL_DIM][CONV1_OUT_CHANNELS];
    static acc_t conv2_acc[BATCH_SIZE * CONV2_OUT_COL_DIM][CONV2_OUT_CHANNELS];
    static acc_t conv3_acc[BATCH_SIZE * CONV3_OUT_COL_DIM][CONV3_OUT_CHANNELS];
    static acc_t fc1_acc[BATCH_SIZE][FC1_OUT_FEATURES];
    static acc_t fc2_acc[BATCH_SIZE][FC2_OUT_FEATURES];

    static elem_t conv1_out[BATCH_SIZE][1][CONV1_OUT_COL_DIM][CONV1_OUT_CHANNELS];
    static elem_t conv2_out[BATCH_SIZE][1][CONV2_OUT_COL_DIM][CONV2_OUT_CHANNELS];
    static elem_t conv3_out[BATCH_SIZE][1][CONV3_OUT_COL_DIM][CONV3_OUT_CHANNELS];

    static elem_t pooled[BATCH_SIZE][FC1_IN_FEATURES];
    static elem_t fc1_out[BATCH_SIZE][FC1_OUT_FEATURES];
    static elem_t fc2_out[BATCH_SIZE][FC2_OUT_FEATURES];

    flatten_conv_weights_1d(CONV1_OUT_CHANNELS, CONV1_IN_CHANNELS, CONV1_KERNEL,
        (const elem_t *)conv1_weight, (elem_t *)conv1_weights_mat);
    flatten_conv_weights_1d(CONV2_OUT_CHANNELS, CONV2_IN_CHANNELS, CONV2_KERNEL,
        (const elem_t *)conv2_weight, (elem_t *)conv2_weights_mat);
    flatten_conv_weights_1d(CONV3_OUT_CHANNELS, CONV3_IN_CHANNELS, CONV3_KERNEL,
        (const elem_t *)conv3_weight, (elem_t *)conv3_weights_mat);
    flatten_fc_weights(FC1_OUT_FEATURES, FC1_IN_FEATURES,
        (const elem_t *)fc1_weight, (elem_t *)fc1_weights_mat);
    flatten_fc_weights(FC2_OUT_FEATURES, FC2_IN_FEATURES,
        (const elem_t *)fc2_weight, (elem_t *)fc2_weights_mat);

    printf("Conv1...\n");
    im2col_1d(BATCH_SIZE, IN_COL_DIM, CONV1_IN_CHANNELS,
        CONV1_KERNEL, CONV1_PADDING, CONV1_STRIDE, CONV1_OUT_COL_DIM,
        (const elem_t *)input, (elem_t *)conv1_im2col);

    tiled_matmul_auto(BATCH_SIZE * CONV1_OUT_COL_DIM, CONV1_OUT_CHANNELS,
        CONV1_KERNEL * CONV1_IN_CHANNELS,
        (elem_t *)conv1_im2col, (elem_t *)conv1_weights_mat,
        (const acc_t *)conv1_bias, (acc_t *)conv1_acc,
        CONV1_KERNEL * CONV1_IN_CHANNELS, CONV1_OUT_CHANNELS,
        CONV1_OUT_CHANNELS, CONV1_OUT_CHANNELS,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int col = 0; col < CONV1_OUT_COL_DIM; col++) {
            const int row = b * CONV1_OUT_COL_DIM + col;
            for (int ch = 0; ch < CONV1_OUT_CHANNELS; ch++) {
                int8_t q = requantize_int8(conv1_acc[row][ch], conv1_M[ch], conv1_zero_points[ch]);
                if (q < 0) {
                    q = 0;
                }
                conv1_out[b][0][col][ch] = q;
            }
        }
    }
    printf("[GEMMINI] conv1 M[0]=%.8f Z[0]=%d\n", conv1_M[0], conv1_zero_points[0]);
    printf("[GEMMINI] conv1 out_row[0..7]: ");
    for (int i = 0; i < 8 && i < CONV1_OUT_CHANNELS; i++) {
        printf("%d ", conv1_out[0][0][0][i]);
    }
    printf("\n");

    printf("Conv2...\n");
    im2col_1d(BATCH_SIZE, CONV1_OUT_COL_DIM, CONV2_IN_CHANNELS,
        CONV2_KERNEL, CONV2_PADDING, CONV2_STRIDE, CONV2_OUT_COL_DIM,
        (const elem_t *)conv1_out, (elem_t *)conv2_im2col);

    tiled_matmul_auto(BATCH_SIZE * CONV2_OUT_COL_DIM, CONV2_OUT_CHANNELS,
        CONV2_KERNEL * CONV2_IN_CHANNELS,
        (elem_t *)conv2_im2col, (elem_t *)conv2_weights_mat,
        (const acc_t *)conv2_bias, (acc_t *)conv2_acc,
        CONV2_KERNEL * CONV2_IN_CHANNELS, CONV2_OUT_CHANNELS,
        CONV2_OUT_CHANNELS, CONV2_OUT_CHANNELS,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int col = 0; col < CONV2_OUT_COL_DIM; col++) {
            const int row = b * CONV2_OUT_COL_DIM + col;
            for (int ch = 0; ch < CONV2_OUT_CHANNELS; ch++) {
                int8_t q = requantize_int8(conv2_acc[row][ch], conv2_M[ch], conv2_zero_points[ch]);
                if (q < 0) {
                    q = 0;
                }
                conv2_out[b][0][col][ch] = q;
            }
        }
    }
    printf("[GEMMINI] conv2 M[0]=%.8f Z[0]=%d\n", conv2_M[0], conv2_zero_points[0]);
    printf("[GEMMINI] conv2 out_row[0..7]: ");
    for (int i = 0; i < 8 && i < CONV2_OUT_CHANNELS; i++) {
        printf("%d ", conv2_out[0][0][0][i]);
    }
    printf("\n");

    printf("Conv3...\n");
    im2col_1d(BATCH_SIZE, CONV2_OUT_COL_DIM, CONV3_IN_CHANNELS,
        CONV3_KERNEL, CONV3_PADDING, CONV3_STRIDE, CONV3_OUT_COL_DIM,
        (const elem_t *)conv2_out, (elem_t *)conv3_im2col);

    tiled_matmul_auto(BATCH_SIZE * CONV3_OUT_COL_DIM, CONV3_OUT_CHANNELS,
        CONV3_KERNEL * CONV3_IN_CHANNELS,
        (elem_t *)conv3_im2col, (elem_t *)conv3_weights_mat,
        (const acc_t *)conv3_bias, (acc_t *)conv3_acc,
        CONV3_KERNEL * CONV3_IN_CHANNELS, CONV3_OUT_CHANNELS,
        CONV3_OUT_CHANNELS, CONV3_OUT_CHANNELS,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int col = 0; col < CONV3_OUT_COL_DIM; col++) {
            const int row = b * CONV3_OUT_COL_DIM + col;
            for (int ch = 0; ch < CONV3_OUT_CHANNELS; ch++) {
                int8_t q = requantize_int8(conv3_acc[row][ch], conv3_M[ch], conv3_zero_points[ch]);
                if (q < 0) {
                    q = 0;
                }
                conv3_out[b][0][col][ch] = q;
            }
        }
    }
    printf("[GEMMINI] conv3 M[0]=%.8f Z[0]=%d\n", conv3_M[0], conv3_zero_points[0]);
    printf("[GEMMINI] conv3 out_row[0..7]: ");
    for (int i = 0; i < 8 && i < CONV3_OUT_CHANNELS; i++) {
        printf("%d ", conv3_out[0][0][0][i]);
    }
    printf("\n");

    printf("Adaptive max pool...\n");
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int ch = 0; ch < CONV3_OUT_CHANNELS; ch++) {
            elem_t max_val = conv3_out[b][0][0][ch];
            for (int col = 1; col < CONV3_OUT_COL_DIM; col++) {
                elem_t v = conv3_out[b][0][col][ch];
                if (v > max_val) {
                    max_val = v;
                }
            }
            pooled[b][ch] = max_val;
        }
    }
    printf("[GEMMINI] pooled[0..7]: ");
    for (int i = 0; i < 8 && i < FC1_IN_FEATURES; i++) {
        printf("%d ", pooled[0][i]);
    }
    printf("\n");

    printf("FC1...\n");
    tiled_matmul_auto(BATCH_SIZE, FC1_OUT_FEATURES, FC1_IN_FEATURES,
        (elem_t *)pooled, (elem_t *)fc1_weights_mat,
        (const acc_t *)fc1_bias, (acc_t *)fc1_acc,
        FC1_IN_FEATURES, FC1_OUT_FEATURES,
        FC1_OUT_FEATURES, FC1_OUT_FEATURES,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int ch = 0; ch < FC1_OUT_FEATURES; ch++) {
            int8_t q = requantize_int8(fc1_acc[b][ch], fc1_M[ch], fc1_zero_points[ch]);
            if (q < 0) {
                q = 0;
            }
            fc1_out[b][ch] = q;
        }
    }
    printf("[GEMMINI] fc1 out[0..7]: ");
    for (int i = 0; i < 8 && i < FC1_OUT_FEATURES; i++) {
        printf("%d ", fc1_out[0][i]);
    }
    printf("\n");

    printf("FC2...\n");
    tiled_matmul_auto(BATCH_SIZE, FC2_OUT_FEATURES, FC2_IN_FEATURES,
        (elem_t *)fc1_out, (elem_t *)fc2_weights_mat,
        (const acc_t *)fc2_bias, (acc_t *)fc2_acc,
        FC2_IN_FEATURES, FC2_OUT_FEATURES,
        FC2_OUT_FEATURES, FC2_OUT_FEATURES,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        true,
        false, false,
        true, false,
        0,
        WS);
    for (int b = 0; b < BATCH_SIZE; b++) {
        fc2_out[b][0] = requantize_int8(fc2_acc[b][0], fc2_M[0], fc2_zero_points[0]);
    }

    for (int b = 0; b < BATCH_SIZE; b++) {
        float logit = (float)fc2_out[b][0] * FC2_OUT_SCALE;
        float prob = sigmoidf(logit);
        printf("Output logit (int8): %d\n", fc2_out[b][0]);
        printf("Output sigmoid: %f\n", prob);
    }

    return 0;
}
