#!/usr/bin/env python3
"""
Run TensorFlow inference on the hard-coded gesture sample (converted from C array).
"""

import argparse
import numpy as np
import tensorflow as tf
import architecture

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


def build_input(layout: str) -> np.ndarray:
    """
    Convert the flattened gesture into a tensor.
    layout == "cw": provided array is [C][W] flattened channel by channel.
    layout == "wc": provided array is [W][C] (time-major).
    """
    if layout == "cw":
        arr = GESTURE_FLOATS.reshape(5, 50)
        arr = np.transpose(arr, (1, 0))  # -> [W, C]
    else:
        arr = GESTURE_FLOATS.reshape(50, 5)

    arr = arr / 360.0  # match training scaling
    arr = arr[np.newaxis, ...]  # [1, W, C]
    return arr


def main():
    parser = argparse.ArgumentParser(description="TF inference on hard-coded gesture sample.")
    parser.add_argument("--model", default="model.h5", help="Path to Keras model (.h5).")
    parser.add_argument("--layout", choices=["wc", "cw"], default="wc",
                        help="Interpretation of the flattened gesture array.")
    parser.add_argument("--show-summary", action="store_true",
                        help="Print model.summary() before running inference.")
    args = parser.parse_args()

    arr = build_input(args.layout)
    custom_objects = {k: v for k, v in architecture.__dict__.items() if callable(v)}
    model = tf.keras.models.load_model(args.model, compile=False, custom_objects=custom_objects)
    if args.show_summary:
        model.summary()

    preds = model(arr, training=False).numpy()[0]
    formatted = ", ".join(f"{x:.6f}" for x in preds)
    print(f"output: [{formatted}]")
    print("argmax:", int(np.argmax(preds)))


if __name__ == "__main__":
    main()
