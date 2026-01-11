# 1D CNN header
Here I use the same naming logic as pytorch to create neural network in C program.

These C programs only supports inference for 1d cnn, but can be extended flexibly 2D conv, 3D conv .. etc

For input, weight, bias, zero point, scale is needed but not strict to the data types. 

and `Input` need to put as 1D `[N][C][H][W]`


`Weight` need to put as 1D [][][]


## parameter
see 1dcnn_param.h

For now, this header only support 

### Activation function type
* RELU
* SIGMOID
* TANH
* LEAKY_RELU
* SOFTMAX
* NONE

### Layer type

* CONV, POOL, Fully connected(FC)

## create layer
First you want to use this header you must 

```C!
CNN* model = createCNN();
```

then

```C!
// create model
CNN* model = createCNN();

// create layer

Layer *conv1 =  nn_Conv1d(...);

// connect layer

addLayer(model, conv1);

...
```

## forward layer
After model creation, layer creation

and connect layers each other to the model

Right now, you need to prepare input data to forward.

```C!
...

forward(inputdata)

...
```

## prepare input


## example of a PTQ


```C!
void sentence_1dcnn() {
    printf("1D CNN Inference Demo\n");
    // init ping-pong buffer
    printf("Initializing ping-pong buffers...\n");
    init_pingpong_buffer(BUFFER_SIZE, ELEM_INT8);
    /* definition layers  */
    printf("Defining layers...\n");
    Layer *conv1 = nn_Conv1d(1, 384, 5, 64, 1, 2, RELU, conv1_weight, conv1_bias, conv1_M, conv1_zero_points, ELEM_INT8);
    Layer *conv2 = nn_Conv1d(64, conv1->outputShape.W, 5, 128, 1, 2, RELU, conv2_weight, conv2_bias, conv2_M, conv2_zero_points, ELEM_INT8);
    Layer *conv3 = nn_Conv1d(128, conv2->outputShape.W, 3, 256, 1, 1, RELU, conv3_weight, conv3_bias, conv3_M, conv3_zero_points, ELEM_INT8);
    Layer *maxpool = nn_AdaptiveMaxPool1d(conv3->outputShape.C, conv3->outputShape.W, 1, ELEM_INT8);
    Layer *fc1 = nn_Linear(maxpool->outputShape.C * maxpool->outputShape.W, 128, NONE, fc1_weight, fc1_bias, fc1_M, fc1_zero_points, ELEM_INT8);
    Layer *fc2 = nn_Linear(128, 1, SIGMOID, fc2_weight, fc2_bias, fc2_M, fc2_zero_points, ELEM_INT8);

    /* connect layers together */
    printf("Creating CNN model...\n");
    CNN* model = createCNN();

    printf("Adding layers to the model...\n");
    addLayer(model, conv1);
    addLayer(model, conv2);
    addLayer(model, conv3);
    addLayer(model, maxpool);
    addLayer(model, fc1);
    addLayer(model, fc2);

    /* input data */
    int8_t *embedding = (int8_t*)random_embedding;

    // inference

    forward(model, embedding);

    // print output
    int8_t *output = (int8_t *) buffer2; // final output in buffer2
    printf("Ground truth (valid_embedding): %d\n", valid_embedding);
    printf("Model prediction (output): %d\n", output[0]);

    if (output[0] == valid_embedding) {
        printf("Prediction correct!\n");
    } else {
        printf("Prediction incorrect!\n");
    }

    if (output[0] == 1)
        printf("→ Model predicts: Valid sentence\n");
    else
        printf("→ Model predicts: Invalid sentence\n");

    freeCNN(model);
    free(buffer1);
    free(buffer2);
}

```
