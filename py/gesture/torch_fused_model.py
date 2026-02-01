#!/usr/bin/env python3
"""
PyTorch inference model that mirrors the C implementation (Conv1d + ReLU, BN fused).

It loads weights from the fused Keras model (e.g., fused_model.h5) and expects
channel-first inputs shaped [N, C, W] where W is the window length.
"""

import argparse
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
# W,C = TIME,CHANNEL
GESTURE_FLOATS = np.array([
    -8, 0, 100, 100, 88,
    -8, 0, 100, 100, 94,
    -8, 0, 100, 100, 99,
    -8, 0, 100, 100, 100,
    -7, 0, 100, 100, 100,
    -7, 0, 100, 100, 100,
    -7, 0, 100, 100, 100,
    -7, 0, 100, 100, 100,
    -6, 0, 100, 100, 100,
    -6, 0, 100, 100, 100,
    -6, 0, 100, 100, 100,
    -8, 0, 100, 100, 100,
    -8, 0, 100, 100, 100,
    -8, 0, 100, 100, 100,
    -8, 0, 100, 100, 100,
    -9, 0, 100, 100, 100,
    -10, 0, 100, 100, 100,
    -11, 0, 100, 100, 100,
    -11, 0, 100, 100, 100,
    -11, 0, 100, 100, 100,
    -12, 0, 100, 100, 100,
    -13, 0, 100, 100, 100,
    -13, 0, 100, 100, 100,
    -14, 0, 100, 100, 100,
    -14, 0, 100, 100, 100,
    -14, 0, 100, 100, 100,
    -14, 0, 100, 100, 100,
    -15, 0, 100, 100, 100,
    -15, 0, 100, 100, 100,
    -15, 0, 100, 100, 97,
    -15, 0, 100, 100, 80,
    -15, 0, 82, 90, 59,
    -14, 0, 61, 43, 44,
    -11, 0, 43, 7, 29,
    -10, 0, 33, 0, 21,
    -10, 0, 22, 0, 13,
    -10, 0, 17, 0, 8,
    -10, 0, 10, 0, 5,
    -10, 0, 4, 0, 4,
    -10, 0, 5, 0, 4,
    -10, 0, 0, 0, 0,
    -10, 0, 0, 0, 0,
    -10, 0, 0, 0, 0,
    -10, 0, 0, 0, 0,
    -10, 0, 0, 0, 0,
    -12, 0, 0, 0, 0,
    -12, 0, 0, 0, 0,
    -13, 0, 0, 0, 0,
    -14, 0, 0, 0, 0,
    -14, 0, 0, 0, 0
], dtype=np.float32)


def _load_custom_objects() -> Optional[Dict[str, object]]:
    """Best-effort import of custom callables so Keras can deserialize the model."""
    try:
        import architecture  # type: ignore
    except Exception:
        return None

    custom: Dict[str, object] = {}
    for name in dir(architecture):
        attr = getattr(architecture, name)
        if callable(attr):
            custom[name] = attr
    return custom or None


class GestureCNN(nn.Module):
    """
    5×50 input, Conv1d stack with valid padding and a final dense layer.
    BN is assumed to be fused into the Conv weights/bias (matching the C side).
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 4, window_size: int = 50):
        super().__init__()
        self.window_size = window_size
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, bias=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, bias=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, bias=True)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, bias=True)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=1, bias=True)

        # Each 3‑tap valid Conv reduces width by 2; the final 1x1 keeps width.
        flattened = 256 * (window_size - 8)
        self.fc = nn.Linear(flattened, num_classes)

    def forward(self, x: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        # Match Keras Flatten order (time-major) by moving width ahead of channels.
        x = torch.flatten(x.transpose(1, 2).contiguous(), start_dim=1)
        logits = self.fc(x)
        print(logits)
        print(torch.softmax(logits, dim=1))
        return torch.softmax(logits, dim=1) if apply_softmax else logits


def _transpose_conv1d(kernel: np.ndarray) -> np.ndarray:
    """Keras Conv1D weights are [K, inC, outC]; PyTorch wants [outC, inC, K]."""
    return np.transpose(kernel, (2, 1, 0)).astype(np.float32)


def _load_conv_weights(torch_layer: nn.Conv1d, keras_layer) -> None:
    kernel, bias = keras_layer.get_weights()
    torch_layer.weight.data.copy_(torch.from_numpy(_transpose_conv1d(kernel)))
    torch_layer.bias.data.copy_(torch.from_numpy(bias.astype(np.float32)))


def _load_dense_weights(torch_layer: nn.Linear, keras_layer) -> None:
    kernel, bias = keras_layer.get_weights()
    # Keras Dense kernel: [in_dim, out_dim]; PyTorch Linear: [out_dim, in_dim]
    torch_layer.weight.data.copy_(torch.from_numpy(kernel.T.astype(np.float32)))
    torch_layer.bias.data.copy_(torch.from_numpy(bias.astype(np.float32)))


def load_fused_weights(
    model: GestureCNN, fused_h5: pathlib.Path, verbose: bool = True
) -> None:
    """
    Load weights from a BN-fused Keras .h5 file into the PyTorch model.

    The loader searches for either `*_fused` layer names (produced by fuse_bn.py)
    or the original unfused names.
    """
    custom_objects = _load_custom_objects()
    keras_model = tf.keras.models.load_model(
        fused_h5, compile=False, custom_objects=custom_objects
    )
    layer_by_name = {layer.name: layer for layer in keras_model.layers}

    def pick(name: str):
        return layer_by_name.get(name) or layer_by_name.get(name.replace("_fused", ""))

    mapping: Tuple[Tuple[str, nn.Module], ...] = (
        ("conv1d_1_fused", model.conv1),
        ("conv1d_2_fused", model.conv2),
        ("conv1d_3_fused", model.conv3),
        ("conv1d_4_fused", model.conv4),
        ("conv1d_5_fused", model.conv5),
    )

    for keras_name, torch_layer in mapping:
        layer = pick(keras_name)
        if layer is None:
            raise KeyError(f"Missing Conv layer {keras_name} in {fused_h5}")
        _load_conv_weights(torch_layer, layer)
        if verbose:
            print(f"✓ loaded {keras_name} -> {torch_layer.__class__.__name__}")

    dense = pick("softmax_1")
    if dense is None:
        raise KeyError("Missing Dense/softmax_1 layer in fused Keras model")
    _load_dense_weights(model.fc, dense)
    if verbose:
        print("✓ loaded softmax_1 -> Linear")


def prepare_input(array: np.ndarray, in_channels: int = 5) -> torch.Tensor:
    """
    Accepts [C, W], [W, C], or [N, C, W] numpy arrays and converts to a
    float32 torch tensor shaped [N, C, W].
    """
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.shape[1] != in_channels and array.shape[2] == in_channels:
        array = np.transpose(array, (0, 2, 1))
    if array.shape[1] != in_channels:
        raise ValueError(f"Expected channel dimension {in_channels}, got {array.shape}")
    return torch.from_numpy(array.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyTorch inference aligned with the C model.")
    parser.add_argument(
        "--weights",
        default="fused_model.h5",
        type=pathlib.Path,
        help="BN-fused Keras .h5 to load (from fuse_bn.py).",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Optional .npy tensor to run (accepts [C,W], [W,C], or [N,C,W]).",
    )
    parser.add_argument("--no-softmax", action="store_true", help="Return logits instead of probabilities.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = GestureCNN()
    load_fused_weights(model, args.weights)
    model.eval()

    if args.input:
        arr = np.load(args.input)
    else:
        # Use the built-in gesture sample (interpreted as [C,W] and scaled)
        arr = GESTURE_FLOATS.reshape(50, 5).T  # -> [C, W]
        arr = arr / 360.0

    x = prepare_input(arr, in_channels=model.conv1.in_channels)
    print(f"Input shape: {x.shape}")
    print(x.numpy())
    with torch.no_grad():
        out = model(x, apply_softmax=not args.no_softmax)
    print(out.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    main()
