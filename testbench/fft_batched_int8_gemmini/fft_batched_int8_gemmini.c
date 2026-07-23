/* SPDX-License-Identifier: Apache-2.0 */
/* Scaled Q7 radix-2 FFT with eight twiddle rotations per Gemmini tile. */

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
static int8_t fused_real[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));
static int8_t fused_imag[FFT_GEMMINI_TILE_DIM][FFT_GEMMINI_BATCHES]
    __attribute__((aligned(64)));

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

int main(void)
{
    uint64_t plan_start = read_cycles();
    make_twiddle_plan();
    uint64_t plan_cycles = read_cycles() - plan_start;
    printf("FFT size: %u\nbatches: %u\nGemmini butterflies/tile: %u\n",
        FFT_INT8_SIZE, FFT_GEMMINI_BATCHES, FFT_GEMMINI_TILE_BUTTERFLIES);
    printf("twiddle plan cycles: %lu\n", (unsigned long)plan_cycles);
    size_t baseline_mismatches = 0, fused_mismatches = 0;
    benchmark_variant("baseline", run_fft, &baseline_mismatches);
    benchmark_variant(
        "stage-fused", run_fft_stage_fused, &fused_mismatches);

    size_t reports = 0;
    for (size_t bin = 0; bin < FFT_INT8_SIZE; ++bin)
        for (size_t batch = 0; batch < FFT_GEMMINI_BATCHES; ++batch) {
            size_t pos = bin * FFT_GEMMINI_BATCHES + batch;
            if (data_real[pos] == fft_int8_groundtruth_real[bin] &&
                data_imag[pos] == fft_int8_groundtruth_imag[bin]) continue;
            if (reports++ < FFT_GEMMINI_MAX_REPORTS)
                printf("mismatch batch %u bin %u: actual %d %dj expected %d %dj\n",
                    (unsigned)batch, (unsigned)bin, data_real[pos], data_imag[pos],
                    fft_int8_groundtruth_real[bin], fft_int8_groundtruth_imag[bin]);
        }
    int ok = baseline_mismatches == 0 && fused_mismatches == 0;
    printf("FFT batched int8 Gemmini: %s (bit-exact Q7 validation)\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
