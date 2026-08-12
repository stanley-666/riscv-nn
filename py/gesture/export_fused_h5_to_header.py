#!/usr/bin/env python3
# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
"""
從 BN 已融合的 Keras .h5 直接導出 C 端使用的權重 header。

輸出權重排列：
  Conv1D: [outC, inC, k] 依序展平，對應 C 端 conv1d_forward 的 oc->ic->k 讀取順序。
  Dense : [out, in] 展平，對應 C 端 fc_forward 的 o->i 讀取順序。
"""

import argparse
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import tensorflow as tf


def _load_custom_objects() -> Optional[Dict[str, object]]:
    """匯入 architecture.py 的自訂符號，避免 load_model 失敗。"""
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


def _fmt_array(arr: np.ndarray) -> str:
    return ",".join(f"{x:.8f}" for x in arr.astype(np.float32).ravel())


def _write_conv(fh, layer, name: str) -> None:
    kernel, bias = layer.get_weights()
    # Keras Conv1D: [k, inC, outC] -> C 期望 [outC, inC, k]
    transposed = np.transpose(kernel, (2, 1, 0))
    flat = transposed.reshape(-1)
    fh.write(f"// Layer: {name}, shape=({transposed.shape[0]}, {transposed.shape[1]}, {transposed.shape[2]})\n")
    fh.write(f"const float {name}_weight[{flat.size}] = {{{_fmt_array(flat)}}};\n\n")
    fh.write(f"// {name}.bias (float32)\n")
    fh.write(f"const float {name}_bias[{bias.size}] = {{{_fmt_array(bias)}}};\n\n")


def _write_dense(fh, layer, name: str) -> None:
    kernel, bias = layer.get_weights()
    # Keras Dense: [in, out] -> C 期望 [out, in]
    transposed = kernel.T
    flat = transposed.reshape(-1)
    fh.write(f"// Dense layer: {name}, shape=({transposed.shape[0]}, {transposed.shape[1]})\n")
    fh.write(f"const float {name}_weight[{flat.size}] = {{{_fmt_array(flat)}}};\n\n")
    fh.write(f"// {name}.bias (float32)\n")
    fh.write(f"const float {name}_bias[{bias.size}] = {{{_fmt_array(bias)}}};\n\n")


def export_header(model_path: pathlib.Path, output_path: pathlib.Path) -> None:
    custom_objects = _load_custom_objects()
    keras_model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    layers = {layer.name: layer for layer in keras_model.layers}

    guard = output_path.stem.upper() + "_H"
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(f"#ifndef {guard}\n#define {guard}\n\n#include <stdint.h>\n\n")

        conv_names: Sequence[str] = [
            "conv1d_1_fused",
            "conv1d_2_fused",
            "conv1d_3_fused",
            "conv1d_4_fused",
            "conv1d_5_fused",
        ]
        for name in conv_names:
            layer = layers.get(name) or layers.get(name.replace("_fused", ""))
            if layer is None:
                raise KeyError(f"找不到 Conv layer: {name}")
            _write_conv(fh, layer, name.replace("_fused", ""))

        dense = layers.get("softmax_1")
        if dense is None:
            raise KeyError("找不到 Dense 層 softmax_1")
        _write_dense(fh, dense, "softmax_1")

        fh.write(f"#endif // {guard}\n")

    print(f"✅ Exported weights to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fused Keras .h5 weights to a C header.")
    parser.add_argument("--model", required=True, help="BN-fused Keras .h5 (e.g., fused_model.h5)")
    parser.add_argument("--output", default="weights_fused_fp32.h", help="Destination header path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_header(pathlib.Path(args.model), pathlib.Path(args.output))


if __name__ == "__main__":
    main()
