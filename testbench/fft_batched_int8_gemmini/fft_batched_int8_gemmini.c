/* SPDX-License-Identifier: Apache-2.0 */
/* Scaled Q7 radix-2 and dense mixed-radix FFT experiments for Gemmini WS. */

#define _DEFAULT_SOURCE

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"
#include "fft_int8_vectors.h"
#include "nn_runtime.h"

#define FFT_GEMMINI_BATCHES 64
#define FFT_GEMMINI_TILE_BUTTERFLIES 8
#define FFT_GEMMINI_TILE_DIM (2 * FFT_GEMMINI_TILE_BUTTERFLIES)
#define FFT_GEMMINI_RUNS 10
#define FFT_GEMMINI_MAX_REPORTS 64
#define FFT_GEMMINI_STAGES 10
#define FFT_GEMMINI_MAX_TILES 513
#define FFT_GEMMINI_FUSED_GROUPS 4
#define FFT_GEMMINI_MAX_FUSED_TILES 512
#define FFT_GEMMINI_RADIX_MAX 16
#define FFT_GEMMINI_RADIX_REAL_DIM (2 * FFT_GEMMINI_RADIX_MAX)
#define FFT_GEMMINI_RADIX_TRANSFORMS 69
#define FFT_GEMMINI_RADIX_MAX_COLUMNS \
    ((FFT_INT8_SIZE / 4) * FFT_GEMMINI_BATCHES)

typedef struct __attribute__((aligned(64))) {
    elem_t weights[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM];
    uint16_t even_bins[FFT_GEMMINI_TILE_BUTTERFLIES];
    uint16_t odd_bins[FFT_GEMMINI_TILE_BUTTERFLIES];
    int8_t wi[FFT_GEMMINI_TILE_BUTTERFLIES];
    uint8_t count;
} gemmini_twiddle_tile_t;

typedef struct __attribute__((aligned(64))) {
    elem_t weights_a[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM];
    elem_t weights_b[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM];
    uint16_t bins[FFT_GEMMINI_FUSED_GROUPS][4];
    int8_t wi_a[FFT_GEMMINI_FUSED_GROUPS];
    int8_t wi_b0[FFT_GEMMINI_FUSED_GROUPS];
    int8_t wi_b1[FFT_GEMMINI_FUSED_GROUPS];
    uint8_t count;
    uint8_t unity;
} gemmini_fused_tile_t;

typedef struct __attribute__((aligned(64))) {
    elem_t weights[FFT_GEMMINI_RADIX_REAL_DIM]
                  [FFT_GEMMINI_RADIX_REAL_DIM];
    uint16_t span;
    uint16_t previous_span;
    uint16_t column_count;
    uint8_t radix;
    uint8_t log2_radix;
    uint8_t offset;
} gemmini_radix_transform_t;

static int8_t data_real[FFT_INT8_SIZE * FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static int8_t data_imag[FFT_INT8_SIZE * FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static int8_t twiddle_real[FFT_INT8_SIZE - 1] __attribute__((aligned(64)));
static int8_t twiddle_imag[FFT_INT8_SIZE - 1] __attribute__((aligned(64)));
static gemmini_twiddle_tile_t twiddle_tiles[FFT_GEMMINI_MAX_TILES]
    __attribute__((aligned(64)));
static uint16_t stage_tile_start[FFT_GEMMINI_STAGES];
static uint16_t stage_tile_count[FFT_GEMMINI_STAGES];
static gemmini_fused_tile_t fused_tiles[FFT_GEMMINI_MAX_FUSED_TILES]
    __attribute__((aligned(64)));
static uint16_t fused_tile_count;
static elem_t matrix_x[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static acc_t matrix_y[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static elem_t native_twiddle_y[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static elem_t native_butterfly_x[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static elem_t native_butterfly_y[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static elem_t native_butterfly_weights
    [FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM]
    __attribute__((aligned(64)));
static int8_t native_reference_real[FFT_INT8_SIZE];
static int8_t native_reference_imag[FFT_INT8_SIZE];
static int8_t fused_real[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static int8_t fused_imag[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static gemmini_radix_transform_t
    radix_transforms[FFT_GEMMINI_RADIX_TRANSFORMS]
    __attribute__((aligned(64)));
static elem_t radix_matrix_x[FFT_GEMMINI_RADIX_REAL_DIM]
                            [FFT_GEMMINI_RADIX_MAX_COLUMNS]
    __attribute__((aligned(64)));
static acc_t radix_matrix_y[FFT_GEMMINI_RADIX_REAL_DIM]
                           [FFT_GEMMINI_RADIX_MAX_COLUMNS]
    __attribute__((aligned(64)));
static int8_t radix_reference_real[FFT_INT8_SIZE];
static int8_t radix_reference_imag[FFT_INT8_SIZE];
static uint16_t radix_transform_count;

static int8_t gemmini_scaled_reference(int32_t value, acc_scale_t scale)
{
    return (int8_t)ACC_SCALE(value, scale);
}

static int32_t rnu_shift(int32_t value, unsigned shift)
{
    return (value + (INT32_C(1) << (shift - 1))) >> shift;
}

static int8_t saturate_i8(int32_t value)
{
    if (value > 127) return 127;
    if (value < -128) return -128;
    return (int8_t)value;
}

static int8_t quantize_q7(float value)
{
    long scaled = lroundf(value * 128.0f);
    if (scaled > 127) scaled = 127;
    if (scaled < -128) scaled = -128;
    return (int8_t)scaled;
}

static unsigned reverse_bits(unsigned value, unsigned bits)
{
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

static void set_complex_weight(
    elem_t weights[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM],
    size_t lane, int8_t wr, int8_t wi)
{
    size_t r = 2 * lane, i = r + 1;
    weights[r][r] = wr;
    weights[r][i] = wi == INT8_MIN ? INT8_MAX : (int8_t)-wi;
    weights[i][r] = wi;
    weights[i][i] = wr;
}

static void make_twiddle_plan(void)
{
    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 1) {
        size_t half = bs >> 1;
        size_t base = half - 1;
        twiddle_real[base] = 127;
        twiddle_imag[base] = 0;
        for (size_t off = 1; off < half; ++off) {
            float angle = -2.0f * (float)M_PI * (float)off / (float)bs;
            twiddle_real[base + off] = quantize_q7(cosf(angle));
            twiddle_imag[base + off] = quantize_q7(sinf(angle));
        }
    }

    size_t next_tile = 0;
    unsigned stage = 0;
    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 1, ++stage) {
        size_t half = bs >> 1;
        size_t stage_base = half - 1;
        size_t total = (FFT_INT8_SIZE / bs) * (half - 1);
        stage_tile_start[stage] = (uint16_t)next_tile;
        stage_tile_count[stage] =
            (uint16_t)((total + FFT_GEMMINI_TILE_BUTTERFLIES - 1) /
                       FFT_GEMMINI_TILE_BUTTERFLIES);
        for (size_t first = 0; first < total;
             first += FFT_GEMMINI_TILE_BUTTERFLIES, ++next_tile) {
            gemmini_twiddle_tile_t *tile = &twiddle_tiles[next_tile];
            size_t count = total - first;
            if (count > FFT_GEMMINI_TILE_BUTTERFLIES)
                count = FFT_GEMMINI_TILE_BUTTERFLIES;
            tile->count = (uint8_t)count;
            for (size_t lane = 0; lane < count; ++lane) {
                size_t id = first + lane;
                size_t block_id = id / (half - 1);
                size_t off = id % (half - 1) + 1;
                size_t even = block_id * bs + off;
                size_t odd = even + half;
                int8_t wr = twiddle_real[stage_base + off];
                int8_t wi = twiddle_imag[stage_base + off];
                size_t r = 2 * lane, i = r + 1;
                tile->even_bins[lane] = (uint16_t)even;
                tile->odd_bins[lane] = (uint16_t)odd;
                tile->wi[lane] = wi;
                tile->weights[r][r] = wr;
                tile->weights[r][i] =
                    wi == INT8_MIN ? INT8_MAX : (int8_t)-wi;
                tile->weights[i][r] = wi;
                tile->weights[i][i] = wr;
            }
        }
    }

    size_t fused_next = 0;
    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 2) {
        size_t half = bs >> 1;
        size_t next_bs = bs << 1;
        size_t stage_a = half - 1;
        size_t stage_b = bs - 1;
        for (size_t off = 0; off < half; ++off) {
            for (size_t first_block = 0; first_block < FFT_INT8_SIZE;
                 first_block += next_bs * FFT_GEMMINI_FUSED_GROUPS) {
                gemmini_fused_tile_t *tile = &fused_tiles[fused_next++];
                size_t remaining =
                    (FFT_INT8_SIZE - first_block + next_bs - 1) / next_bs;
                if (remaining > FFT_GEMMINI_FUSED_GROUPS)
                    remaining = FFT_GEMMINI_FUSED_GROUPS;
                tile->count = (uint8_t)remaining;
                tile->unity = off == 0;
                int8_t ar = twiddle_real[stage_a + off];
                int8_t ai = twiddle_imag[stage_a + off];
                int8_t b0r = twiddle_real[stage_b + off];
                int8_t b0i = twiddle_imag[stage_b + off];
                int8_t b1r = twiddle_real[stage_b + half + off];
                int8_t b1i = twiddle_imag[stage_b + half + off];
                for (size_t group = 0; group < remaining; ++group) {
                    size_t block = first_block + group * next_bs;
                    tile->bins[group][0] = (uint16_t)(block + off);
                    tile->bins[group][1] =
                        (uint16_t)(block + half + off);
                    tile->bins[group][2] =
                        (uint16_t)(block + bs + off);
                    tile->bins[group][3] =
                        (uint16_t)(block + bs + half + off);
                    tile->wi_a[group] = ai;
                    tile->wi_b0[group] = b0i;
                    tile->wi_b1[group] = b1i;
                    set_complex_weight(tile->weights_a, 2 * group, ar, ai);
                    set_complex_weight(
                        tile->weights_a, 2 * group + 1, ar, ai);
                    if (off != 0)
                        set_complex_weight(
                            tile->weights_b, 2 * group, b0r, b0i);
                    set_complex_weight(
                        tile->weights_b, 2 * group + 1, b1r, b1i);
                }
            }
        }
    }
    fused_tile_count = (uint16_t)fused_next;

    static const uint8_t radices[] = {4, 16, 16};
    static const uint8_t log2_radices[] = {2, 4, 4};
    size_t radix_next = 0;
    size_t previous_span = 1;
    for (size_t stage = 0; stage < 3; ++stage) {
        size_t radix = radices[stage];
        size_t span = previous_span * radix;
        size_t blocks = FFT_INT8_SIZE / span;
        size_t columns = blocks * FFT_GEMMINI_BATCHES;
        for (size_t offset = 0; offset < previous_span; ++offset) {
            gemmini_radix_transform_t *transform =
                &radix_transforms[radix_next++];
            transform->span = (uint16_t)span;
            transform->previous_span = (uint16_t)previous_span;
            transform->column_count = (uint16_t)columns;
            transform->radix = (uint8_t)radix;
            transform->log2_radix = log2_radices[stage];
            transform->offset = (uint8_t)offset;
            for (size_t output = 0; output < radix; ++output)
                for (size_t input = 0; input < radix; ++input) {
                    float angle =
                        -2.0f * (float)M_PI * (float)input *
                        (float)(output * previous_span + offset) /
                        (float)span;
                    int8_t wr = quantize_q7(cosf(angle));
                    int8_t wi = quantize_q7(sinf(angle));
                    transform->weights[output][input] = wr;
                    transform->weights[output][radix + input] =
                        quantize_q7(-sinf(angle));
                    transform->weights[radix + output][input] = wi;
                    transform->weights[radix + output][radix + input] = wr;
                }
        }
        previous_span = span;
    }
    radix_transform_count = (uint16_t)radix_next;

    for (size_t group = 0; group < FFT_GEMMINI_FUSED_GROUPS; ++group) {
        size_t ar = 4 * group, tr = ar + 1;
        size_t ai = ar + 2, ti = ar + 3;
        native_butterfly_weights[ar][ar] = 1;
        native_butterfly_weights[ar][tr] = 1;
        native_butterfly_weights[tr][ar] = 1;
        native_butterfly_weights[tr][tr] = -1;
        native_butterfly_weights[ai][ai] = 1;
        native_butterfly_weights[ai][ti] = 1;
        native_butterfly_weights[ti][ai] = 1;
        native_butterfly_weights[ti][ti] = -1;
    }
}

static size_t mixed_radix_source_index(size_t destination)
{
    size_t digit0 = destination & 3;
    size_t digit1 = (destination >> 2) & 15;
    size_t digit2 = destination >> 6;
    return digit0 * 256 + digit1 * 16 + digit2;
}

static void mixed_radix_permute(void)
{
    static int8_t permuted_real[FFT_INT8_SIZE * FFT_GEMMINI_BATCHES]
        __attribute__((aligned(64)));
    static int8_t permuted_imag[FFT_INT8_SIZE * FFT_GEMMINI_BATCHES]
        __attribute__((aligned(64)));
    for (size_t destination = 0; destination < FFT_INT8_SIZE; ++destination) {
        size_t source = mixed_radix_source_index(destination);
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t destination_pos =
                destination * FFT_GEMMINI_BATCHES + batch;
            size_t source_pos = source * FFT_GEMMINI_BATCHES + batch;
            permuted_real[destination_pos] = data_real[source_pos];
            permuted_imag[destination_pos] = data_imag[source_pos];
        }
    }
    for (size_t pos = 0; pos < FFT_INT8_SIZE * FFT_GEMMINI_BATCHES; ++pos) {
        data_real[pos] = permuted_real[pos];
        data_imag[pos] = permuted_imag[pos];
    }
}

static void make_mixed_radix_reference(void)
{
    for (size_t destination = 0; destination < FFT_INT8_SIZE; ++destination) {
        size_t source = mixed_radix_source_index(destination);
        radix_reference_real[destination] = fft_int8_input_real[source];
        radix_reference_imag[destination] = fft_int8_input_imag[source];
    }

    for (size_t index = 0; index < radix_transform_count; ++index) {
        const gemmini_radix_transform_t *transform =
            &radix_transforms[index];
        size_t radix = transform->radix;
        size_t span = transform->span;
        size_t previous_span = transform->previous_span;
        size_t offset = transform->offset;
        for (size_t block = 0; block < FFT_INT8_SIZE; block += span) {
            int8_t output_real[FFT_GEMMINI_RADIX_MAX];
            int8_t output_imag[FFT_GEMMINI_RADIX_MAX];
            for (size_t output = 0; output < radix; ++output) {
                int32_t real_acc = 0, imag_acc = 0;
                for (size_t input = 0; input < radix; ++input) {
                    size_t bin = block + offset + input * previous_span;
                    int32_t br = radix_reference_real[bin];
                    int32_t bi = radix_reference_imag[bin];
                    real_acc += transform->weights[output][input] * br;
                    real_acc +=
                        transform->weights[output][radix + input] * bi;
                    imag_acc +=
                        transform->weights[radix + output][input] * br;
                    imag_acc +=
                        transform->weights[radix + output][radix + input] * bi;
                }
                unsigned shift = 7 + transform->log2_radix;
                output_real[output] =
                    saturate_i8(rnu_shift(real_acc, shift));
                output_imag[output] =
                    saturate_i8(rnu_shift(imag_acc, shift));
            }
            for (size_t output = 0; output < radix; ++output) {
                size_t bin = block + offset + output * previous_span;
                radix_reference_real[bin] = output_real[output];
                radix_reference_imag[bin] = output_imag[output];
            }
        }
    }
}

static void run_fft_mixed_radix(void)
{
    mixed_radix_permute();
    for (size_t index = 0; index < radix_transform_count; ++index) {
        const gemmini_radix_transform_t *transform =
            &radix_transforms[index];
        size_t radix = transform->radix;
        size_t span = transform->span;
        size_t previous_span = transform->previous_span;
        size_t offset = transform->offset;
        size_t columns = transform->column_count;
        size_t column = 0;
        for (size_t block = 0; block < FFT_INT8_SIZE; block += span)
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES;
                 ++batch, ++column)
                for (size_t input = 0; input < radix; ++input) {
                    size_t bin = block + offset + input * previous_span;
                    size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
                    radix_matrix_x[input][column] = data_real[pos];
                    radix_matrix_x[radix + input][column] = data_imag[pos];
                }

        size_t real_dim = 2 * radix;
        tiled_matmul_auto(real_dim, columns, real_dim,
            &transform->weights[0][0], &radix_matrix_x[0][0], NULL,
            &radix_matrix_y[0][0], FFT_GEMMINI_RADIX_REAL_DIM,
            FFT_GEMMINI_RADIX_MAX_COLUMNS, FFT_GEMMINI_RADIX_MAX_COLUMNS,
            FFT_GEMMINI_RADIX_MAX_COLUMNS,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, false, true, false, 0, WS);

        column = 0;
        for (size_t block = 0; block < FFT_INT8_SIZE; block += span)
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES;
                 ++batch, ++column)
                for (size_t output = 0; output < radix; ++output) {
                    size_t bin = block + offset + output * previous_span;
                    size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
                    unsigned shift = 7 + transform->log2_radix;
                    data_real[pos] = saturate_i8(
                        rnu_shift(radix_matrix_y[output][column], shift));
                    data_imag[pos] = saturate_i8(rnu_shift(
                        radix_matrix_y[radix + output][column], shift));
                }
    }
}

static void make_native_reference(void)
{
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin) {
        size_t reversed = reverse_bits((unsigned)bin, 10);
        native_reference_real[reversed] = fft_int8_input_real[bin];
        native_reference_imag[reversed] = fft_int8_input_imag[bin];
    }

    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 1) {
        size_t half = bs >> 1;
        size_t stage_base = half - 1;
        for (size_t block = 0; block < FFT_INT8_SIZE; block += bs)
            for (size_t off = 0; off < half; ++off) {
                size_t even = block + off, odd = even + half;
                int32_t ar = native_reference_real[even];
                int32_t ai = native_reference_imag[even];
                int32_t br = native_reference_real[odd];
                int32_t bi = native_reference_imag[odd];
                int8_t tr, ti;
                if (off == 0) {
                    tr = (int8_t)br;
                    ti = (int8_t)bi;
                } else {
                    int32_t wr = twiddle_real[stage_base + off];
                    int32_t wi = twiddle_imag[stage_base + off];
                    tr = gemmini_scaled_reference(
                        wr * br - wi * bi, 1.0f / 128.0f);
                    ti = gemmini_scaled_reference(
                        wi * br + wr * bi, 1.0f / 128.0f);
                }
                native_reference_real[even] =
                    gemmini_scaled_reference(ar + tr, 0.5f);
                native_reference_imag[even] =
                    gemmini_scaled_reference(ai + ti, 0.5f);
                native_reference_real[odd] =
                    gemmini_scaled_reference(ar - tr, 0.5f);
                native_reference_imag[odd] =
                    gemmini_scaled_reference(ai - ti, 0.5f);
            }
    }
}

static void load_input(void)
{
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
            data_real[pos] = fft_int8_input_real[bin];
            data_imag[pos] = fft_int8_input_imag[bin];
        }
}

static void bit_reverse(void)
{
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin) {
        size_t reversed = reverse_bits((unsigned)bin, 10);
        if (reversed <= bin) continue;
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t a = bin * FFT_GEMMINI_BATCHES + batch;
            size_t b = reversed * FFT_GEMMINI_BATCHES + batch;
            int8_t t = data_real[a]; data_real[a] = data_real[b]; data_real[b] = t;
            t = data_imag[a]; data_imag[a] = data_imag[b]; data_imag[b] = t;
        }
    }
}

static void unity_butterflies(size_t bs)
{
    size_t half = bs >> 1;
    for (size_t block = 0; block < FFT_INT8_SIZE; block += bs)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t e = block * FFT_GEMMINI_BATCHES + batch;
            size_t o = (block + half) * FFT_GEMMINI_BATCHES + batch;
            int32_t er = data_real[e], ei = data_imag[e];
            int32_t br = data_real[o], bi = data_imag[o];
            data_real[e] = (int8_t)rnu_shift(er + br, 1);
            data_imag[e] = (int8_t)rnu_shift(ei + bi, 1);
            data_real[o] = (int8_t)rnu_shift(er - br, 1);
            data_imag[o] = (int8_t)rnu_shift(ei - bi, 1);
        }
}

static void clear_input_tile(void)
{
    for (size_t i = 0; i < FFT_GEMMINI_TILE_DIM; ++i)
        for (size_t b = 0; b < FFT_GEMMINI_BATCHES; ++b) matrix_x[i][b] = 0;
}

static void gemmini_stage(unsigned stage)
{
    size_t begin = stage_tile_start[stage];
    size_t end = begin + stage_tile_count[stage];
    for (size_t tile_index = begin; tile_index < end; ++tile_index) {
        const gemmini_twiddle_tile_t *tile = &twiddle_tiles[tile_index];
        size_t count = tile->count;
        clear_input_tile();
        for (size_t lane = 0; lane < count; ++lane) {
            size_t odd = tile->odd_bins[lane];
            size_t r = 2 * lane, i = r + 1;
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                size_t pos = odd * FFT_GEMMINI_BATCHES + batch;
                matrix_x[r][batch] = data_real[pos];
                matrix_x[i][batch] = data_imag[pos];
            }
        }
        tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
            FFT_GEMMINI_TILE_DIM, &tile->weights[0][0], &matrix_x[0][0], NULL,
            &matrix_y[0][0], FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
            FFT_GEMMINI_BATCHES, FFT_GEMMINI_BATCHES,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            false, false, false, true, false, 0, WS);
        for (size_t lane = 0; lane < count; ++lane) {
            size_t r = 2 * lane, i = r + 1;
            size_t even = tile->even_bins[lane], odd = tile->odd_bins[lane];
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                size_t e = even * FFT_GEMMINI_BATCHES + batch;
                size_t o = odd * FFT_GEMMINI_BATCHES + batch;
                int32_t real_acc = matrix_y[r][batch];
                if (tile->wi[lane] == INT8_MIN) real_acc += data_imag[o];
                int8_t tr = saturate_i8(rnu_shift(real_acc, 7));
                int8_t ti = saturate_i8(rnu_shift(matrix_y[i][batch], 7));
                int32_t er = data_real[e], ei = data_imag[e];
                data_real[e] = (int8_t)rnu_shift(er + tr, 1);
                data_imag[e] = (int8_t)rnu_shift(ei + ti, 1);
                data_real[o] = (int8_t)rnu_shift(er - tr, 1);
                data_imag[o] = (int8_t)rnu_shift(ei - ti, 1);
            }
        }
    }
}

static void run_fft(void)
{
    bit_reverse();
    unsigned stage = 0;
    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 1, ++stage) {
        unity_butterflies(bs);
        if (stage_tile_count[stage] != 0) gemmini_stage(stage);
    }
}

static void native_butterfly_apply(
    size_t count, const uint16_t *even_bins, const uint16_t *odd_bins,
    size_t twiddle_lane, int unity)
{
    for (size_t row = 0; row < FFT_GEMMINI_TILE_DIM; ++row)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch)
            native_butterfly_x[row][batch] = 0;

    for (size_t group = 0; group < count; ++group) {
        size_t ar = 4 * group, tr = ar + 1;
        size_t ai = ar + 2, ti = ar + 3;
        size_t even_base = even_bins[group] * FFT_GEMMINI_BATCHES;
        size_t odd_base = odd_bins[group] * FFT_GEMMINI_BATCHES;
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            native_butterfly_x[ar][batch] = data_real[even_base + batch];
            native_butterfly_x[ai][batch] = data_imag[even_base + batch];
            if (unity) {
                native_butterfly_x[tr][batch] = data_real[odd_base + batch];
                native_butterfly_x[ti][batch] = data_imag[odd_base + batch];
            } else {
                native_butterfly_x[tr][batch] =
                    native_twiddle_y[2 * (twiddle_lane + group)][batch];
                native_butterfly_x[ti][batch] =
                    native_twiddle_y[2 * (twiddle_lane + group) + 1][batch];
            }
        }
    }

    tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
        FFT_GEMMINI_TILE_DIM, &native_butterfly_weights[0][0],
        &native_butterfly_x[0][0], NULL, &native_butterfly_y[0][0],
        FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
        FFT_GEMMINI_BATCHES, FFT_GEMMINI_BATCHES,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 0.5f, 0,
        false, false, false, false, false, 0, WS);

    for (size_t group = 0; group < count; ++group) {
        size_t ar = 4 * group, lower_r = ar + 1;
        size_t ai = ar + 2, lower_i = ar + 3;
        size_t even_base = even_bins[group] * FFT_GEMMINI_BATCHES;
        size_t odd_base = odd_bins[group] * FFT_GEMMINI_BATCHES;
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            data_real[even_base + batch] = native_butterfly_y[ar][batch];
            data_imag[even_base + batch] = native_butterfly_y[ai][batch];
            data_real[odd_base + batch] = native_butterfly_y[lower_r][batch];
            data_imag[odd_base + batch] = native_butterfly_y[lower_i][batch];
        }
    }
}

static void native_unity_butterflies(size_t bs)
{
    uint16_t even_bins[FFT_GEMMINI_FUSED_GROUPS];
    uint16_t odd_bins[FFT_GEMMINI_FUSED_GROUPS];
    size_t half = bs >> 1, count = 0;
    for (size_t block = 0; block < FFT_INT8_SIZE; block += bs) {
        even_bins[count] = (uint16_t)block;
        odd_bins[count] = (uint16_t)(block + half);
        if (++count == FFT_GEMMINI_FUSED_GROUPS) {
            native_butterfly_apply(count, even_bins, odd_bins, 0, 1);
            count = 0;
        }
    }
    if (count != 0)
        native_butterfly_apply(count, even_bins, odd_bins, 0, 1);
}

static void gemmini_stage_native(unsigned stage)
{
    size_t begin = stage_tile_start[stage];
    size_t end = begin + stage_tile_count[stage];
    for (size_t tile_index = begin; tile_index < end; ++tile_index) {
        const gemmini_twiddle_tile_t *tile = &twiddle_tiles[tile_index];
        size_t count = tile->count;
        clear_input_tile();
        for (size_t lane = 0; lane < count; ++lane) {
            size_t odd = tile->odd_bins[lane];
            size_t r = 2 * lane, i = r + 1;
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                size_t pos = odd * FFT_GEMMINI_BATCHES + batch;
                matrix_x[r][batch] = data_real[pos];
                matrix_x[i][batch] = data_imag[pos];
            }
        }

        tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
            FFT_GEMMINI_TILE_DIM, &tile->weights[0][0], &matrix_x[0][0], NULL,
            &native_twiddle_y[0][0], FFT_GEMMINI_TILE_DIM,
            FFT_GEMMINI_BATCHES, FFT_GEMMINI_BATCHES, FFT_GEMMINI_BATCHES,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, 1.0f / 128.0f, 0,
            false, false, false, false, false, 0, WS);

        for (size_t lane = 0; lane < count; ++lane) {
            if (tile->wi[lane] != INT8_MIN) continue;
            size_t odd_base = tile->odd_bins[lane] * FFT_GEMMINI_BATCHES;
            size_t r = 2 * lane, i = r + 1;
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                int32_t br = data_real[odd_base + batch];
                int32_t bi = data_imag[odd_base + batch];
                int32_t wr = tile->weights[r][r];
                int32_t wi = tile->wi[lane];
                native_twiddle_y[r][batch] =
                    gemmini_scaled_reference(wr * br - wi * bi,
                                              1.0f / 128.0f);
                native_twiddle_y[i][batch] =
                    gemmini_scaled_reference(wi * br + wr * bi,
                                              1.0f / 128.0f);
            }
        }

        for (size_t first = 0; first < count;
             first += FFT_GEMMINI_FUSED_GROUPS) {
            size_t group_count = count - first;
            if (group_count > FFT_GEMMINI_FUSED_GROUPS)
                group_count = FFT_GEMMINI_FUSED_GROUPS;
            native_butterfly_apply(group_count,
                &tile->even_bins[first], &tile->odd_bins[first], first, 0);
        }
    }
}

static void run_fft_gemmini_native(void)
{
    bit_reverse();
    unsigned stage = 0;
    for (size_t bs = 2; bs <= FFT_INT8_SIZE; bs <<= 1, ++stage) {
        native_unity_butterflies(bs);
        if (stage_tile_count[stage] != 0) gemmini_stage_native(stage);
    }
}

static void gemmini_matmul(
    const elem_t weights[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM])
{
    tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
        FFT_GEMMINI_TILE_DIM, &weights[0][0], &matrix_x[0][0], NULL,
        &matrix_y[0][0], FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_BATCHES,
        FFT_GEMMINI_BATCHES, FFT_GEMMINI_BATCHES,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        false, false, false, true, false, 0, WS);
}

static void run_fft_stage_fused(void)
{
    bit_reverse();
    for (size_t tile_index = 0; tile_index < fused_tile_count; ++tile_index) {
        const gemmini_fused_tile_t *tile = &fused_tiles[tile_index];
        size_t count = tile->count;
        clear_input_tile();

        if (!tile->unity) {
            for (size_t group = 0; group < count; ++group) {
                size_t p1 = tile->bins[group][1] * FFT_GEMMINI_BATCHES;
                size_t p3 = tile->bins[group][3] * FFT_GEMMINI_BATCHES;
                size_t lane0 = 2 * group, lane1 = lane0 + 1;
                size_t r0 = 2 * lane0, i0 = r0 + 1;
                size_t r1 = 2 * lane1, i1 = r1 + 1;
                for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                    matrix_x[r0][batch] = data_real[p1 + batch];
                    matrix_x[i0][batch] = data_imag[p1 + batch];
                    matrix_x[r1][batch] = data_real[p3 + batch];
                    matrix_x[i1][batch] = data_imag[p3 + batch];
                }
            }
            gemmini_matmul(tile->weights_a);
        }

        for (size_t group = 0; group < count; ++group) {
            size_t lane0 = 2 * group, lane1 = lane0 + 1;
            size_t r0 = 2 * lane0, i0 = r0 + 1;
            size_t r1 = 2 * lane1, i1 = r1 + 1;
            size_t bins[4] = {tile->bins[group][0], tile->bins[group][1],
                              tile->bins[group][2], tile->bins[group][3]};
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                size_t p0 = bins[0] * FFT_GEMMINI_BATCHES + batch;
                size_t p1 = bins[1] * FFT_GEMMINI_BATCHES + batch;
                size_t p2 = bins[2] * FFT_GEMMINI_BATCHES + batch;
                size_t p3 = bins[3] * FFT_GEMMINI_BATCHES + batch;
                int8_t t1r, t1i, t3r, t3i;
                if (tile->unity) {
                    t1r = data_real[p1]; t1i = data_imag[p1];
                    t3r = data_real[p3]; t3i = data_imag[p3];
                } else {
                    int32_t a1r = matrix_y[r0][batch];
                    int32_t a3r = matrix_y[r1][batch];
                    if (tile->wi_a[group] == INT8_MIN) {
                        a1r += data_imag[p1];
                        a3r += data_imag[p3];
                    }
                    t1r = saturate_i8(rnu_shift(a1r, 7));
                    t1i = saturate_i8(rnu_shift(matrix_y[i0][batch], 7));
                    t3r = saturate_i8(rnu_shift(a3r, 7));
                    t3i = saturate_i8(rnu_shift(matrix_y[i1][batch], 7));
                }
                int32_t r0v = data_real[p0], i0v = data_imag[p0];
                int32_t r2v = data_real[p2], i2v = data_imag[p2];
                fused_real[4 * group][batch] =
                    (int8_t)rnu_shift(r0v + t1r, 1);
                fused_imag[4 * group][batch] =
                    (int8_t)rnu_shift(i0v + t1i, 1);
                fused_real[4 * group + 1][batch] =
                    (int8_t)rnu_shift(r0v - t1r, 1);
                fused_imag[4 * group + 1][batch] =
                    (int8_t)rnu_shift(i0v - t1i, 1);
                fused_real[4 * group + 2][batch] =
                    (int8_t)rnu_shift(r2v + t3r, 1);
                fused_imag[4 * group + 2][batch] =
                    (int8_t)rnu_shift(i2v + t3i, 1);
                fused_real[4 * group + 3][batch] =
                    (int8_t)rnu_shift(r2v - t3r, 1);
                fused_imag[4 * group + 3][batch] =
                    (int8_t)rnu_shift(i2v - t3i, 1);
            }
        }

        clear_input_tile();
        for (size_t group = 0; group < count; ++group) {
            size_t lane0 = 2 * group, lane1 = lane0 + 1;
            size_t r0 = 2 * lane0, i0 = r0 + 1;
            size_t r1 = 2 * lane1, i1 = r1 + 1;
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                if (!tile->unity) {
                    matrix_x[r0][batch] = fused_real[4 * group + 2][batch];
                    matrix_x[i0][batch] = fused_imag[4 * group + 2][batch];
                }
                matrix_x[r1][batch] = fused_real[4 * group + 3][batch];
                matrix_x[i1][batch] = fused_imag[4 * group + 3][batch];
            }
        }
        gemmini_matmul(tile->weights_b);

        for (size_t group = 0; group < count; ++group) {
            size_t lane0 = 2 * group, lane1 = lane0 + 1;
            size_t r0 = 2 * lane0, i0 = r0 + 1;
            size_t r1 = 2 * lane1, i1 = r1 + 1;
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                int8_t q0r, q0i;
                if (tile->unity) {
                    q0r = fused_real[4 * group + 2][batch];
                    q0i = fused_imag[4 * group + 2][batch];
                } else {
                    int32_t value = matrix_y[r0][batch];
                    if (tile->wi_b0[group] == INT8_MIN)
                        value += fused_imag[4 * group + 2][batch];
                    q0r = saturate_i8(rnu_shift(value, 7));
                    q0i = saturate_i8(rnu_shift(matrix_y[i0][batch], 7));
                }
                int32_t value = matrix_y[r1][batch];
                if (tile->wi_b1[group] == INT8_MIN)
                    value += fused_imag[4 * group + 3][batch];
                int8_t q1r = saturate_i8(rnu_shift(value, 7));
                int8_t q1i =
                    saturate_i8(rnu_shift(matrix_y[i1][batch], 7));
                int32_t y0r = fused_real[4 * group][batch];
                int32_t y0i = fused_imag[4 * group][batch];
                int32_t y1r = fused_real[4 * group + 1][batch];
                int32_t y1i = fused_imag[4 * group + 1][batch];
                size_t p0 =
                    tile->bins[group][0] * FFT_GEMMINI_BATCHES + batch;
                size_t p1 =
                    tile->bins[group][1] * FFT_GEMMINI_BATCHES + batch;
                size_t p2 =
                    tile->bins[group][2] * FFT_GEMMINI_BATCHES + batch;
                size_t p3 =
                    tile->bins[group][3] * FFT_GEMMINI_BATCHES + batch;
                data_real[p0] = (int8_t)rnu_shift(y0r + q0r, 1);
                data_imag[p0] = (int8_t)rnu_shift(y0i + q0i, 1);
                data_real[p2] = (int8_t)rnu_shift(y0r - q0r, 1);
                data_imag[p2] = (int8_t)rnu_shift(y0i - q0i, 1);
                data_real[p1] = (int8_t)rnu_shift(y1r + q1r, 1);
                data_imag[p1] = (int8_t)rnu_shift(y1i + q1i, 1);
                data_real[p3] = (int8_t)rnu_shift(y1r - q1r, 1);
                data_imag[p3] = (int8_t)rnu_shift(y1i - q1i, 1);
            }
        }
    }
}

static size_t count_mismatches(void)
{
    size_t mismatches = 0;
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
            if (data_real[pos] != fft_int8_groundtruth_real[bin] ||
                data_imag[pos] != fft_int8_groundtruth_imag[bin])
                ++mismatches;
        }
    return mismatches;
}

static int validate_gemmini_scaling(void)
{
    static elem_t a[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM]
        __attribute__((aligned(64)));
    static elem_t b[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM]
        __attribute__((aligned(64)));
    static elem_t c[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_TILE_DIM]
        __attribute__((aligned(64)));
    static const int8_t butterfly_inputs[FFT_GEMMINI_TILE_DIM][2] = {
        {1, 0}, {2, 1}, {-2, -1}, {127, 127},
        {-128, -128}, {127, -128}, {-128, 127}, {1, -2},
        {-1, 2}, {5, 0}, {-5, 0}, {126, 127},
        {-127, -128}, {64, 63}, {-64, -63}, {0, -128},
    };

    for (size_t row = 0; row < FFT_GEMMINI_TILE_DIM; ++row)
        for (size_t col = 0; col < FFT_GEMMINI_TILE_DIM; ++col) {
            a[row][col] = (int8_t)(((row * 17 + col * 29) % 255) - 128);
            b[row][col] = (int8_t)(((row * 31 + col * 13) % 255) - 128);
            c[row][col] = 0;
        }
    for (size_t k = 0; k < FFT_GEMMINI_TILE_DIM; ++k) a[0][k] = 0;
    a[0][0] = 1;
    b[0][0] = 64;
    b[0][1] = -64;

    tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        FFT_GEMMINI_TILE_DIM, &a[0][0], &b[0][0], NULL, &c[0][0],
        FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 1.0f / 128.0f, 0,
        false, false, false, false, false, 0, WS);

    size_t matmul_native_mismatches = 0, matmul_rnu_differences = 0;
    for (size_t row = 0; row < FFT_GEMMINI_TILE_DIM; ++row)
        for (size_t col = 0; col < FFT_GEMMINI_TILE_DIM; ++col) {
            int32_t sum = 0;
            for (size_t k = 0; k < FFT_GEMMINI_TILE_DIM; ++k)
                sum += (int32_t)a[row][k] * b[k][col];
            int8_t native = gemmini_scaled_reference(sum, 1.0f / 128.0f);
            int8_t rnu = saturate_i8(rnu_shift(sum, 7));
            if (c[row][col] != native) ++matmul_native_mismatches;
            if (native != rnu) ++matmul_rnu_differences;
        }

    for (size_t row = 0; row < FFT_GEMMINI_TILE_DIM; ++row)
        for (size_t col = 0; col < FFT_GEMMINI_TILE_DIM; ++col) {
            a[row][col] = 0;
            b[row][col] = 0;
            c[row][col] = 0;
        }
    for (size_t pair = 0; pair < FFT_GEMMINI_TILE_BUTTERFLIES; ++pair) {
        size_t upper = 2 * pair, lower = upper + 1;
        a[upper][upper] = 1;
        a[upper][lower] = 1;
        a[lower][upper] = 1;
        a[lower][lower] = -1;
    }
    for (size_t col = 0; col < FFT_GEMMINI_TILE_DIM; ++col) {
        for (size_t pair = 0; pair < FFT_GEMMINI_TILE_BUTTERFLIES; ++pair) {
            b[2 * pair][col] = butterfly_inputs[col][0];
            b[2 * pair + 1][col] = butterfly_inputs[col][1];
        }
    }

    tiled_matmul_auto(FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        FFT_GEMMINI_TILE_DIM, &a[0][0], &b[0][0], NULL, &c[0][0],
        FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        FFT_GEMMINI_TILE_DIM, FFT_GEMMINI_TILE_DIM,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, 0.5f, 0,
        false, false, false, false, false, 0, WS);

    size_t butterfly_native_mismatches = 0, butterfly_rnu_differences = 0;
    for (size_t col = 0; col < FFT_GEMMINI_TILE_DIM; ++col) {
        int32_t input_a = butterfly_inputs[col][0];
        int32_t input_t = butterfly_inputs[col][1];
        int8_t native_upper =
            gemmini_scaled_reference(input_a + input_t, 0.5f);
        int8_t native_lower =
            gemmini_scaled_reference(input_a - input_t, 0.5f);
        int8_t rnu_upper = (int8_t)rnu_shift(input_a + input_t, 1);
        int8_t rnu_lower = (int8_t)rnu_shift(input_a - input_t, 1);
        for (size_t pair = 0; pair < FFT_GEMMINI_TILE_BUTTERFLIES; ++pair) {
            if (c[2 * pair][col] != native_upper ||
                c[2 * pair + 1][col] != native_lower)
                ++butterfly_native_mismatches;
        }
        if (native_upper != rnu_upper || native_lower != rnu_lower)
            ++butterfly_rnu_differences;
    }

    printf("Gemmini scaling validation:\n");
    printf("  matmul native-reference mismatches: %u / %u\n",
        (unsigned)matmul_native_mismatches,
        FFT_GEMMINI_TILE_DIM * FFT_GEMMINI_TILE_DIM);
    printf("  matmul native-vs-RNU differences: %u / %u\n",
        (unsigned)matmul_rnu_differences,
        FFT_GEMMINI_TILE_DIM * FFT_GEMMINI_TILE_DIM);
    printf("  butterfly native-reference mismatches: %u / %u\n",
        (unsigned)butterfly_native_mismatches,
        FFT_GEMMINI_TILE_DIM * FFT_GEMMINI_TILE_BUTTERFLIES);
    printf("  butterfly native-vs-RNU cases: %u / %u\n",
        (unsigned)butterfly_rnu_differences, FFT_GEMMINI_TILE_DIM);

    return matmul_native_mismatches == 0 &&
           butterfly_native_mismatches == 0;
}

static uint64_t benchmark_variant(const char *name, void (*function)(void),
                                  size_t *mismatches)
{
    uint64_t total = 0;
    gemmini_flush(0);
    for (unsigned run = 0; run < FFT_GEMMINI_RUNS; ++run) {
        load_input();
        uint64_t start = read_cycles();
        function();
        total += read_cycles() - start;
    }
    *mismatches = count_mismatches();
    uint64_t average = total / FFT_GEMMINI_RUNS;
    printf("\nGemmini variant: %s\nbenchmark runs: %u\n", name,
        FFT_GEMMINI_RUNS);
    printf("average FFT cycles: %lu\n", (unsigned long)average);
    printf("mismatched complex points: %u / %u\n", (unsigned)*mismatches,
        FFT_INT8_SIZE * FFT_GEMMINI_BATCHES);
    printf("FFT int8 Gemmini %s: %s\n", name,
        *mismatches == 0 ? "PASS" : "FAIL");
    return average;
}

static uint64_t benchmark_native_variant(size_t *mismatches)
{
    uint64_t total = 0;
    gemmini_flush(0);
    for (unsigned run = 0; run < FFT_GEMMINI_RUNS; ++run) {
        load_input();
        uint64_t start = read_cycles();
        run_fft_gemmini_native();
        total += read_cycles() - start;
    }

    *mismatches = 0;
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
            if (data_real[pos] != native_reference_real[bin] ||
                data_imag[pos] != native_reference_imag[bin])
                ++*mismatches;
        }

    uint64_t average = total / FFT_GEMMINI_RUNS;
    printf("\nGemmini variant: native-scaling\nbenchmark runs: %u\n",
        FFT_GEMMINI_RUNS);
    printf("average FFT cycles: %lu\n", (unsigned long)average);
    printf("native-reference mismatched complex points: %u / %u\n",
        (unsigned)*mismatches, FFT_INT8_SIZE * FFT_GEMMINI_BATCHES);
    printf("FFT int8 Gemmini native-scaling: %s\n",
        *mismatches == 0 ? "PASS" : "FAIL");
    return average;
}

static uint64_t benchmark_mixed_radix_variant(
    size_t *reference_mismatches, size_t *radix2_mismatches)
{
    uint64_t total = 0;
    gemmini_flush(0);
    for (unsigned run = 0; run < FFT_GEMMINI_RUNS; ++run) {
        load_input();
        uint64_t start = read_cycles();
        run_fft_mixed_radix();
        total += read_cycles() - start;
    }

    *reference_mismatches = 0;
    *radix2_mismatches = 0;
    uint64_t absolute_error_sum = 0;
    size_t within_one = 0;
    unsigned max_component_error = 0;
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
            if (data_real[pos] != radix_reference_real[bin] ||
                data_imag[pos] != radix_reference_imag[bin])
                ++*reference_mismatches;
            if (data_real[pos] != fft_int8_groundtruth_real[bin] ||
                data_imag[pos] != fft_int8_groundtruth_imag[bin])
                ++*radix2_mismatches;
            int real_error =
                (int)data_real[pos] - fft_int8_groundtruth_real[bin];
            int imag_error =
                (int)data_imag[pos] - fft_int8_groundtruth_imag[bin];
            if (real_error < 0) real_error = -real_error;
            if (imag_error < 0) imag_error = -imag_error;
            absolute_error_sum += (unsigned)real_error + (unsigned)imag_error;
            if (real_error <= 1 && imag_error <= 1) ++within_one;
            if ((unsigned)real_error > max_component_error)
                max_component_error = (unsigned)real_error;
            if ((unsigned)imag_error > max_component_error)
                max_component_error = (unsigned)imag_error;
        }

    uint64_t average = total / FFT_GEMMINI_RUNS;
    printf("\nGemmini variant: mixed-radix-4-16-16\n");
    printf("benchmark runs: %u\n", FFT_GEMMINI_RUNS);
    printf("dense transform calls per FFT: %u\n",
        (unsigned)radix_transform_count);
    printf("average FFT cycles: %lu\n", (unsigned long)average);
    printf("mixed-radix-reference mismatched complex points: %u / %u\n",
        (unsigned)*reference_mismatches,
        FFT_INT8_SIZE * FFT_GEMMINI_BATCHES);
    printf("radix-2-groundtruth differing complex points: %u / %u\n",
        (unsigned)*radix2_mismatches,
        FFT_INT8_SIZE * FFT_GEMMINI_BATCHES);
    printf("radix-2-groundtruth points within +/-1 per component: %u / %u\n",
        (unsigned)within_one, FFT_INT8_SIZE * FFT_GEMMINI_BATCHES);
    printf("radix-2-groundtruth mean absolute component error x1000: %lu\n",
        (unsigned long)(absolute_error_sum * 1000 /
            (2 * FFT_INT8_SIZE * FFT_GEMMINI_BATCHES)));
    printf("radix-2-groundtruth maximum component error: %u\n",
        max_component_error);
    printf("FFT int8 Gemmini mixed-radix-4-16-16: %s\n",
        *reference_mismatches == 0 ? "PASS" : "FAIL");
    return average;
}

int main(void)
{
    uint64_t plan_start = read_cycles();
    make_twiddle_plan();
    uint64_t plan_cycles = read_cycles() - plan_start;
    printf("FFT size: %u\nbatches: %u\nGemmini butterflies/tile: %u\n",
        FFT_INT8_SIZE, FFT_GEMMINI_BATCHES, FFT_GEMMINI_TILE_BUTTERFLIES);
    printf("twiddle plan cycles: %lu\n", (unsigned long)plan_cycles);
    make_native_reference();
    make_mixed_radix_reference();
    size_t baseline_mismatches = 0, fused_mismatches = 0;
    size_t native_mismatches = 0;
    size_t radix_reference_mismatches = 0, radix2_differences = 0;
    benchmark_variant("baseline", run_fft, &baseline_mismatches);
    benchmark_mixed_radix_variant(
        &radix_reference_mismatches, &radix2_differences);
    benchmark_variant(
        "stage-fused", run_fft_stage_fused, &fused_mismatches);
    benchmark_native_variant(&native_mismatches);
    int scaling_ok = validate_gemmini_scaling();

    if (fused_mismatches != 0) {
        load_input();
        run_fft_stage_fused();
        size_t reports = 0;
        for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
            for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
                size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
                if (data_real[pos] == fft_int8_groundtruth_real[bin] &&
                    data_imag[pos] == fft_int8_groundtruth_imag[bin]) continue;
                if (reports++ < FFT_GEMMINI_MAX_REPORTS)
                    printf("mismatch batch %u bin %u: actual %d %dj "
                           "expected %d %dj\n",
                        (unsigned)batch, (unsigned)bin,
                        data_real[pos], data_imag[pos],
                        fft_int8_groundtruth_real[bin],
                        fft_int8_groundtruth_imag[bin]);
            }
    }
    int ok = scaling_ok && baseline_mismatches == 0 &&
             fused_mismatches == 0 &&
             native_mismatches == 0 && radix_reference_mismatches == 0;
    printf("FFT batched int8 Gemmini: %s "
           "(Q7 and native-scaling validation)\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
