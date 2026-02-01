#!/usr/bin/env python3
"""
Fuse Conv1D + BatchNormalization pairs inside a Keras model so the exported
model no longer depends on BN operators (useful for bare‑metal inference).

Usage:
    python fuse_bn.py --model-path model.h5 --output-path fused_model.h5
"""

import argparse
import pathlib
from typing import Dict, Optional

import numpy as np
import tensorflow as tf


def _load_custom_objects() -> Optional[Dict[str, object]]:
    """
    Import custom items from architecture.py when available so load_model works.
    """
    try:
        import architecture  # pylint: disable=import-error
    except Exception:  # pragma: no cover - best effort import
        return None

    custom = {}
    for name in dir(architecture):
        attr = getattr(architecture, name)
        if callable(attr):
            custom[name] = attr
    return custom or None


def fuse_conv_bn_pair(conv_layer, bn_layer):
    """Return weights for a Conv1D layer after absorbing BN statistics."""
    kernel, *rest = conv_layer.get_weights()
    bias = rest[0] if rest else np.zeros(kernel.shape[-1], dtype=kernel.dtype)

    gamma = bn_layer.gamma.numpy()
    beta = bn_layer.beta.numpy()
    moving_mean = bn_layer.moving_mean.numpy()
    moving_var = bn_layer.moving_variance.numpy()
    epsilon = bn_layer.epsilon

    denom = np.sqrt(moving_var + epsilon)
    scale = gamma / denom
    fused_kernel = kernel * scale
    fused_bias = (bias - moving_mean) * scale + beta
    return [fused_kernel, fused_bias]


def build_fused_model(model: tf.keras.Model) -> tf.keras.Model:
    """Clone the model, replacing Conv1D+BN pairs with single Conv1D."""
    if len(model.inputs) != 1:
        raise ValueError("Only single-input models are supported.")
    input_tensor = model.inputs[0]
    input_name = input_tensor.name.split(":")[0]
    inp = tf.keras.Input(shape=input_tensor.shape[1:], name=input_name)
    prev_tensor = inp

    layers = model.layers
    idx = 0
    while idx < len(layers):
        layer = layers[idx]
        if isinstance(layer, tf.keras.layers.InputLayer):
            idx += 1
            continue

        next_layer = layers[idx + 1] if idx + 1 < len(layers) else None
        if (
            isinstance(layer, tf.keras.layers.Conv1D)
            and isinstance(next_layer, tf.keras.layers.BatchNormalization)
        ):
            config = layer.get_config()
            config["use_bias"] = True
            config["name"] = f"{layer.name}_fused"
            fused_conv = tf.keras.layers.Conv1D.from_config(config)
            prev_tensor = fused_conv(prev_tensor)
            fused_conv.set_weights(fuse_conv_bn_pair(layer, next_layer))
            idx += 2
            continue

        cloned = layer.__class__.from_config(layer.get_config())
        prev_tensor = cloned(prev_tensor)
        if layer.get_weights():
            cloned.set_weights(layer.get_weights())
        idx += 1

    fused = tf.keras.Model(inputs=inp, outputs=prev_tensor, name=f"{model.name}_bn_fused")
    return fused


def parse_args():
    parser = argparse.ArgumentParser(description="Fuse Conv1D + BN layers.")
    parser.add_argument("--model-path", required=True, help="Input .h5 or SavedModel directory.")
    parser.add_argument("--output-path", required=True, help="Where to save the fused model.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = pathlib.Path(args.model_path)
    output_path = pathlib.Path(args.output_path)

    custom_objects = _load_custom_objects()
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    fused_model = build_fused_model(model)
    fused_model.save(output_path)
    print(f"Saved BN-fused model to {output_path}")


if __name__ == "__main__":
    main()
