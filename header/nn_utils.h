#ifndef _NN_UTILS_H_
#define _NN_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h> 
#include "nn_param.h"

size_t sizeof_dtype(elem_type t);
void *safe_malloc(size_t size); // if malloc fail, exit program
void *safe_calloc(size_t num, size_t size) ; // if calloc fail, exit program
void safe_free(void *ptr);
/* Convolution 1D  RVV */

float   activate_f32(float x, ActivationType type);
int32_t activate_i32(int32_t x, ActivationType type);
int8_t  activate_i8(int8_t x, ActivationType type);

void softmax_f32(const float *input, float *output, int len);

void *padded_input_create_nchw(NNModule *layer, const void *input) ;
void *padded_input_create_nhwc(NNModule *layer, const void *input) ;
void *input_im2col_create_nhwc_1d(NNModule *layer, const void *input);
void *input_im2col_create_nhwc_2d(NNModule *layer, const void *input);
void *padded_input_create_nchw_2d(NNModule *layer, const void *input);
void *padded_input_create_nhwc_2d(NNModule *layer, const void *input);
// clipping
int8_t clip_i32_i8(int32_t x, int8_t min_val, int8_t max_val);
int8_t requantize_int8_symmetric(const int32_t input, float scale);
int8_t requantize_int8_asymmetric(const int32_t input, float scale, int32_t zero_point);
void init_pingpong_buffer(size_t num_elem, elem_type dtype);
void free_pingpong_buffer();

#endif // _1DCNN_UTILS_H_

/*
Copyright (c) 2025, MC2 Lab, National Taiwan Normal University.All rights reserved.
Author : Stanley Lee
*/
