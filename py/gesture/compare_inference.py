#!/usr/bin/env python3
"""
Compare TensorFlow (architecture.py model) and torch_fused_model inference on the same input.

The script loads the BN-fused Keras weights (fused_model.h5) for both frameworks,
feeds either a provided .npy tensor or the built-in gesture sample, and reports
whether the probability vectors match within the requested tolerance.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf
import torch

import architecture
from torch_fused_model import (
    GESTURE_FLOATS,
    GestureCNN,
    load_fused_weights,
    prepare_input,
)


def _collect_custom_objects() -> Dict[str, object]:
    """Gather callables from architecture.py for tf.keras load_model."""
    return {name: attr for name, attr in architecture.__dict__.items() if callable(attr)}


def _build_default_input(layout: str) -> np.ndarray:
    """Recreate the built-in gesture sample, matching tf_run_sample.py behavior."""
    if layout == "cw":
        arr = GESTURE_FLOATS.reshape(5, 50).transpose(1, 0)
    else:
        arr = GESTURE_FLOATS.reshape(50, 5)
    return arr.astype(np.float32)


def _prepare_tf_input(args: argparse.Namespace) -> np.ndarray:
    """
    Load the requested input tensor, normalize, and ensure [N, W, C] layout.
    Torch input is derived from this array via channel/width transpose.
    """
    if args.input:
        arr = np.load(args.input).astype(np.float32)
    else:
        arr = _build_default_input(args.layout)

    if args.scale:
        arr = arr / args.scale

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor [batch, window, channel], got shape {arr.shape}")
    if arr.shape[-1] != args.channels:
        raise ValueError(
            f"Channel mismatch: got {arr.shape[-1]} channels, expected {args.channels}"
        )
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorFlow vs. Torch output comparison helper.")
    parser.add_argument(
        "--tf-model",
        type=pathlib.Path,
        default=pathlib.Path("fused_model.h5"),
        help="Path to the BN-fused Keras .h5 used for TensorFlow inference.",
    )
    parser.add_argument(
        "--torch-weights",
        type=pathlib.Path,
        default=pathlib.Path("fused_model.h5"),
        help="Path to the BN-fused Keras .h5 used to load Torch weights.",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Optional .npy tensor shaped [batch, window, channel] (default: built-in gesture).",
    )
    parser.add_argument(
        "--layout",
        choices=("wc", "cw"),
        default="wc",
        help="Interpretation of the built-in gesture sample (ignored when --input is set).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=360.0,
        help="Divide inputs by this value (set to 0 to skip scaling).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=5,
        help="Channel count expected by both models.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance used when comparing outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the comparison summary and exit code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf_input = _prepare_tf_input(args)

    window_size = tf_input.shape[1]
    custom_objects = _collect_custom_objects()

    tf_model = tf.keras.models.load_model(
        args.tf_model, compile=False, custom_objects=custom_objects
    )
    tf_out = tf_model(tf_input, training=False).numpy()

    num_classes = tf_out.shape[-1]
    torch_model = GestureCNN(
        in_channels=args.channels, num_classes=num_classes, window_size=window_size
    )
    load_fused_weights(torch_model, args.torch_weights)
    torch_model.eval()

    torch_input = np.transpose(tf_input, (0, 2, 1))
    torch_tensor = prepare_input(torch_input, in_channels=args.channels)

    with torch.no_grad():
        torch_logits = torch_model(torch_tensor, apply_softmax=False)
        torch_out = torch.softmax(torch_logits, dim=1).cpu().numpy()

    diff = np.abs(tf_out - torch_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    argmax_match = np.array_equal(np.argmax(tf_out, axis=1), np.argmax(torch_out, axis=1))

    if not args.quiet:
        flat = torch_tensor.squeeze(0).numpy().reshape(-1)
        print("Input (channel-first) as C initializer:")
        print("{")
        for idx, value in enumerate(flat):
            end = "," if idx < flat.size - 1 else ""
            sep = "\n" if (idx + 1) % 10 == 0 else ""
            print(f"    {value:.6f}{end}", end=sep or " ")
        if flat.size % 10:
            print()
        print("}")
        np.set_printoptions(precision=6, suppress=True)

        def _tf_layer_output(name: str) -> np.ndarray:
            layer_model = tf.keras.Model(tf_model.input, tf_model.get_layer(name).output)
            out = layer_model(tf_input, training=False).numpy()[0]  # [W, C]
            return out.T  # -> [C, W]

        layer_names = ["relu_1", "relu_2", "relu_3", "relu_4", "relu_5"]
        tf_layers = {name: _tf_layer_output(name) for name in layer_names}

        with torch.no_grad():
            torch_layers = []
            x = torch_tensor
            for conv in (torch_model.conv1, torch_model.conv2, torch_model.conv3, torch_model.conv4, torch_model.conv5):
                x = torch.relu(conv(x))
                torch_layers.append(x.squeeze(0).cpu().numpy())

        for name, torch_act in zip(layer_names, torch_layers):
            tf_act = tf_layers[name]
            diff_layer = np.abs(tf_act - torch_act)
            print(f"{name}: TF first channel first 20:", tf_act[0, :20])
            print(f"{name}: Torch first channel first 20:", torch_act[0, :20])
            print(f"{name}: max abs diff={float(diff_layer.max()):.6e}")

        np.set_printoptions(precision=6, suppress=True)
        print("TensorFlow probs:", tf_out)
        print("Torch probs     :", torch_out)
        print("Abs diff        :", diff)
        print(f"Argmax TF: {np.argmax(tf_out, axis=1)}  Torch: {np.argmax(torch_out, axis=1)}")

    print(f"mean|diff|={mean_diff:.6e}, max|diff|={max_diff:.6e}, argmax match={argmax_match}")
    if max_diff > args.atol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
