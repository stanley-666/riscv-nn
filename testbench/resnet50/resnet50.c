#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nn_layer.h"
#include "nn_utils.h"
#include "nn_infer_vpu.h"
#include "resnet50_weights.h"
#include "resnet50_io.h"

#define RESNET_INPUT_H 224
#define RESNET_INPUT_W 224
#define RESNET_INPUT_C 3
#define RESNET_NUM_CLASSES 1000
#define RESNET_SKIP_MAX_ELEMS (56 * 56 * 256)
#define RESNET_MAX_ELEMS (112 * 112 * 64)

static float buffer1_static[RESNET_MAX_ELEMS] __attribute__((aligned(64)));
static float buffer2_static[RESNET_MAX_ELEMS] __attribute__((aligned(64)));

void init_pingpong_buffer(size_t num_elem, elem_type dtype) {
    (void)num_elem;
    (void)dtype;
    size_t size = RESNET_MAX_ELEMS * sizeof(float);
    buffer1 = buffer1_static;
    buffer2 = buffer2_static;
    printf("pingpong buffers (static): buffer1=%p buffer2=%p bytes=%zu\n", buffer1, buffer2, size);
    memset(buffer1, 0, size);
    memset(buffer2, 0, size);
}

NNModule *add_bottleneck(CNN *model,
                                NNModule *input,
                                int midC,
                                int outC,
                                int stride,
                                const float *w1,
                                const float *b1,
                                const float *w2,
                                const float *b2,
                                const float *w3,
                                const float *b3,
                                const float *down_w,
                                const float *down_b,
                                int use_downsample,
                                void *skip_a,
                                void *skip_b)
{
    // Limitations: stride applied on conv2, NHWC input for VPU path, batch=1.
    NNModule *save_in = nn_Save(input->outputShape.C, input->outputShape.H, input->outputShape.W, ELEM_FLOAT32);
    save_in->params.save.buffer = skip_a;

    NNModule *conv1 = nn_Conv2d(input->outputShape.C, input->outputShape.H, input->outputShape.W,
                               1, midC, 1, 0, RELU, w1, b1, NULL, NULL, ELEM_FLOAT32);

    NNModule *conv2 = nn_Conv2d(midC, conv1->outputShape.H, conv1->outputShape.W,
                               3, midC, stride, 1, RELU, w2, b2, NULL, NULL, ELEM_FLOAT32);

    NNModule *conv3 = nn_Conv2d(midC, conv2->outputShape.H, conv2->outputShape.W,
                               1, outC, 1, 0, NONE, w3, b3, NULL, NULL, ELEM_FLOAT32);

    addLayer(model, save_in);
    addLayer(model, conv1);    
    addLayer(model, conv2);                   
    addLayer(model, conv3);

    if (use_downsample) {
        NNModule *save_main = nn_Save(conv3->outputShape.C, conv3->outputShape.H, conv3->outputShape.W, ELEM_FLOAT32);
        save_main->params.save.buffer = skip_b;
        NNModule *down = nn_Conv2d(input->outputShape.C, input->outputShape.H, input->outputShape.W, 1, outC, stride, 0, NONE, down_w, down_b, NULL, NULL, ELEM_FLOAT32);
        down->params.conv2d.input_override = save_in->params.save.buffer;
        NNModule *add = nn_Add(outC, down->outputShape.H, down->outputShape.W, RELU, skip_b, ELEM_FLOAT32);
        
        addLayer(model, save_main);
        addLayer(model, down);
        addLayer(model, add);
        return add;
    } else {
        NNModule *add = nn_Add(outC, conv3->outputShape.H, conv3->outputShape.W, RELU, skip_a, ELEM_FLOAT32);
        addLayer(model, add);
        return add;
    }
}

void resnet50_inference(void) {
    printf("ResNet50 Inference Demo (RVV)\n");
    printf("Initializing ping-pong buffers...\n");
    init_pingpong_buffer(RESNET_MAX_ELEMS, ELEM_FLOAT32);
    size_t shared_skip_bytes = RESNET_SKIP_MAX_ELEMS * sizeof(float);
    void *shared_skip_a = safe_malloc(shared_skip_bytes);
    void *shared_skip_b = safe_malloc(shared_skip_bytes);

    CNN *model = createCNN();

    NNModule *conv1 = nn_Conv2d(RESNET_INPUT_C, RESNET_INPUT_H, RESNET_INPUT_W, 7, 64, 2, 3, RELU, conv1_weight, conv1_bias, NULL, NULL, ELEM_FLOAT32);
    NNModule *maxpool = nn_Pool2d(64, conv1->outputShape.H, conv1->outputShape.W, 3, 2, 1, MAX_POOL, ELEM_FLOAT32);
    NNModule *x = maxpool;
    addLayer(model, conv1);
    addLayer(model, maxpool);
    x = add_bottleneck(model, x, 64, 256, 1,
                       layer1_0_conv1_weight, layer1_0_conv1_bias,
                       layer1_0_conv2_weight, layer1_0_conv2_bias,
                       layer1_0_conv3_weight, layer1_0_conv3_bias,
                       layer1_0_downsample_0_weight, layer1_0_downsample_0_bias, 1,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 64, 256, 1,
                       layer1_1_conv1_weight, layer1_1_conv1_bias,
                       layer1_1_conv2_weight, layer1_1_conv2_bias,
                       layer1_1_conv3_weight, layer1_1_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 64, 256, 1,
                       layer1_2_conv1_weight, layer1_2_conv1_bias,
                       layer1_2_conv2_weight, layer1_2_conv2_bias,
                       layer1_2_conv3_weight, layer1_2_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);

    x = add_bottleneck(model, x, 128, 512, 2,
                       layer2_0_conv1_weight, layer2_0_conv1_bias,
                       layer2_0_conv2_weight, layer2_0_conv2_bias,
                       layer2_0_conv3_weight, layer2_0_conv3_bias,
                       layer2_0_downsample_0_weight, layer2_0_downsample_0_bias, 1,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 128, 512, 1,
                       layer2_1_conv1_weight, layer2_1_conv1_bias,
                       layer2_1_conv2_weight, layer2_1_conv2_bias,
                       layer2_1_conv3_weight, layer2_1_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 128, 512, 1,
                       layer2_2_conv1_weight, layer2_2_conv1_bias,
                       layer2_2_conv2_weight, layer2_2_conv2_bias,
                       layer2_2_conv3_weight, layer2_2_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 128, 512, 1,
                       layer2_3_conv1_weight, layer2_3_conv1_bias,
                       layer2_3_conv2_weight, layer2_3_conv2_bias,
                       layer2_3_conv3_weight, layer2_3_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);

    x = add_bottleneck(model, x, 256, 1024, 2,
                       layer3_0_conv1_weight, layer3_0_conv1_bias,
                       layer3_0_conv2_weight, layer3_0_conv2_bias,
                       layer3_0_conv3_weight, layer3_0_conv3_bias,
                       layer3_0_downsample_0_weight, layer3_0_downsample_0_bias, 1,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 256, 1024, 1,
                       layer3_1_conv1_weight, layer3_1_conv1_bias,
                       layer3_1_conv2_weight, layer3_1_conv2_bias,
                       layer3_1_conv3_weight, layer3_1_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 256, 1024, 1,
                       layer3_2_conv1_weight, layer3_2_conv1_bias,
                       layer3_2_conv2_weight, layer3_2_conv2_bias,
                       layer3_2_conv3_weight, layer3_2_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 256, 1024, 1,
                       layer3_3_conv1_weight, layer3_3_conv1_bias,
                       layer3_3_conv2_weight, layer3_3_conv2_bias,
                       layer3_3_conv3_weight, layer3_3_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 256, 1024, 1,
                       layer3_4_conv1_weight, layer3_4_conv1_bias,
                       layer3_4_conv2_weight, layer3_4_conv2_bias,
                       layer3_4_conv3_weight, layer3_4_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 256, 1024, 1,
                       layer3_5_conv1_weight, layer3_5_conv1_bias,
                       layer3_5_conv2_weight, layer3_5_conv2_bias,
                       layer3_5_conv3_weight, layer3_5_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);

    x = add_bottleneck(model, x, 512, 2048, 2,
                       layer4_0_conv1_weight, layer4_0_conv1_bias,
                       layer4_0_conv2_weight, layer4_0_conv2_bias,
                       layer4_0_conv3_weight, layer4_0_conv3_bias,
                       layer4_0_downsample_0_weight, layer4_0_downsample_0_bias, 1,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 512, 2048, 1,
                       layer4_1_conv1_weight, layer4_1_conv1_bias,
                       layer4_1_conv2_weight, layer4_1_conv2_bias,
                       layer4_1_conv3_weight, layer4_1_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);
    x = add_bottleneck(model, x, 512, 2048, 1,
                       layer4_2_conv1_weight, layer4_2_conv1_bias,
                       layer4_2_conv2_weight, layer4_2_conv2_bias,
                       layer4_2_conv3_weight, layer4_2_conv3_bias,
                       NULL, NULL, 0,
                       shared_skip_a, shared_skip_b);

    NNModule *avgpool = nn_AdaptiveAvgPool2d(2048, x->outputShape.H, x->outputShape.W, 1, 1, ELEM_FLOAT32);  
    NNModule *fc = nn_Linear(2048, RESNET_NUM_CLASSES, NONE, fc_weight, fc_bias, NULL, NULL, ELEM_FLOAT32);
    addLayer(model, avgpool);
    addLayer(model, fc);

    printf("Build ResNet50 complete...\n");
    printf("Starting inference...\n");
    // VPU path expects NHWC input layout; source test vector is NCHW.
    size_t input_elems = RESNET_INPUT_H * RESNET_INPUT_W * RESNET_INPUT_C;
    float *input_nhwc = (float *)safe_malloc(input_elems * sizeof(float));
    for (int h = 0; h < RESNET_INPUT_H; ++h) {
        for (int w = 0; w < RESNET_INPUT_W; ++w) {
            for (int c = 0; c < RESNET_INPUT_C; ++c) {
                size_t nchw_idx = ((0 * RESNET_INPUT_C + c) * RESNET_INPUT_H + h) * RESNET_INPUT_W + w;
                size_t nhwc_idx = (h * RESNET_INPUT_W + w) * RESNET_INPUT_C + c;
                input_nhwc[nhwc_idx] = resnet50_input[nchw_idx];
            }
        }
    }

    forward_input_bytes = input_elems * sizeof(float);
    clock_t start = clock();
    forward_fp32_vpu(model, (void*)input_nhwc);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("ResNet50 inference time: %.6f seconds\n", elapsed);

    float *out_ptr = (float *)((model->numModules % 2 == 0) ? buffer1 : buffer2);
    int max_idx = 0;
    float max_val = out_ptr[0];
    for (int i = 1; i < RESNET_NUM_CLASSES; ++i) {
        if (out_ptr[i] > max_val) {
            max_val = out_ptr[i];
            max_idx = i;
        }
    }
    printf("Top1: class=%d prob=%.6f\n", max_idx, max_val);
    for (int i = 0; i < RESNET_NUM_CLASSES; ++i) {
        printf("logit[%d]=%.9f\n", i, out_ptr[i]);
    }

    safe_free(shared_skip_a);
    safe_free(shared_skip_b);
    safe_free(input_nhwc);
    freeCNN(model);
}

int main(void)
{
    resnet50_inference();
    return 0;
}
