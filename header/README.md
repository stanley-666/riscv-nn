# NN Header API (C)

This folder defines the public C API for building and running CNN graphs
on RISC-V. The APIs mirror common PyTorch naming where possible.

## Supported Layers

From `header/nn_layer.h`:

- `nn_Conv1d`
- `nn_Conv2d`
- `nn_Pool1d` (max/avg)
- `nn_Pool2d` (max/avg)
- `nn_AdaptiveMaxPool1d`
- `nn_AdaptiveAvgPool2d`
- `nn_Linear`
- `nn_Transpose`
- `nn_Save` / `nn_Add` (residual branches)

## Supported Activations

From `header/nn_param.h`:

- `RELU`, `SIGMOID`, `TANH`, `LEAKY_RELU`, `SOFTMAX`, `NONE`

## Data Layout

All tensors use NCHW layout:

- `N`: batch size (currently assumed to be 1 in kernels)
- `C`: channels
- `H`: height (for 1D models, use `H=1`)
- `W`: width / length

Input buffers and outputs are 1D contiguous arrays in NCHW order. For 1D
models the layout is `[C][W]` (since `N=1, H=1`).

Weights are stored as contiguous arrays:

- Conv1d: `[outC][inC][K]`
- Conv2d: `[outC][inC][K][K]` (square kernels)
- Linear: `[outDim][inDim]`

Bias is `[outC]` for conv and `[outDim]` for linear.

## Quantization Parameters

For `ELEM_INT8`, `M` and `zps` are required and are applied per output channel:

- `M`: scale factors (typically `scale_in * scale_w / scale_out`)
- `zps`: output/activation zero points

For `ELEM_FLOAT32`, these can be `NULL`.

### Quantization Limitations

- Only output/activation zero points are supported.
- Weight zero points are not supported (weights are assumed symmetric, `zp=0`).
- Input zero points are not used.

## Limitations

- Inference only (no training).
- Batch size is effectively `N=1` in current kernels.
- Conv2d is limited to square kernels/stride/padding and no dilation.
- Kernel implementations primarily support `ELEM_INT8` and `ELEM_FLOAT32`.
  Other element types are declared but not fully implemented.

## Minimal Usage

```c
CNN *model = createCNN();
NNModule *conv1 = nn_Conv1d(...);
addLayer(model, conv1);
forward(model, input);
```
