#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(__file__).with_name("trace_uv_local_byte_model.pt")
DEFAULT_SAMPLE = Path(__file__).with_name("c_infer_sample_msg0_byte0.json")


def fmt_shape(shape: Any) -> str:
    try:
        return str(tuple(int(x) for x in shape))
    except Exception:
        return str(shape)


def guess_layout(shape: tuple[int, ...]) -> str:
    if len(shape) == 4:
        return "possible Conv2d weight: OIHW / (out_channels, in_channels, kernel_h, kernel_w)"
    if len(shape) == 3:
        return "possible Conv1d weight: OIC / (out_channels, in_channels, kernel)"
    if len(shape) == 2:
        return "possible Linear weight: OI / (out_features, in_features)"
    if len(shape) == 1:
        return "possible bias / norm / embedding vector"
    return "layout not inferred"


def inspect_archive(model_path: Path) -> None:
    print("== File ==")
    print(f"path: {model_path}")
    print(f"size_bytes: {model_path.stat().st_size}")
    is_zip = zipfile.is_zipfile(model_path)
    print(f"is_zip_archive: {is_zip}")
    if not is_zip:
        return

    with zipfile.ZipFile(model_path) as zf:
        names = zf.namelist()
        print("archive_entries:")
        for name in names[:80]:
            print(f"  {name}")
        if len(names) > 80:
            print(f"  ... ({len(names) - 80} more)")

        has_code = any(name.endswith(".py") or "/code/" in name for name in names)
        has_constants = any("constants" in name for name in names)
        has_data_pkl = any(name.endswith("data.pkl") for name in names)
        print("archive_guess:")
        print(f"  contains_data_pkl: {has_data_pkl}")
        print(f"  contains_code: {has_code}")
        print(f"  contains_constants: {has_constants}")
        if has_data_pkl and not has_code:
            print("  likely_format: torch.save checkpoint/state_dict or eager module serialization")
        elif has_data_pkl and has_code:
            print("  likely_format: TorchScript or packaged scripted module")


def inspect_sample_json(sample_path: Path) -> None:
    if not sample_path.exists():
        return
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    print("== Sample Metadata ==")
    for key in ("description", "checkpoint", "message_idx", "byte_idx", "target_groundtruth", "prediction_argmax"):
        if key in data:
            print(f"{key}: {data[key]}")
    if "input_shapes" in data:
        print("input_shapes:")
        for name, shape in data["input_shapes"].items():
            print(f"  {name}: {tuple(shape)}")
    if "top5" in data:
        print("top5:")
        for item in data["top5"]:
            print(f"  class={item['class']} logit={item['logit']}")


def print_state_dict(sd: dict[str, Any]) -> None:
    print("== State Dict ==")
    for name, tensor in sd.items():
        shape = tuple(int(x) for x in tensor.shape)
        print(f"{name}: shape={shape}, dtype={tensor.dtype}, layout={guess_layout(shape)}")


def inspect_with_torch(model_path: Path) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("== Torch ==")
        print("torch not installed in this environment; skipping live tensor inspection.")
        return

    print("== Torch ==")
    loaded: Any = None
    load_mode = None
    jit_error = None

    try:
        loaded = torch.jit.load(str(model_path), map_location="cpu")
        load_mode = "torch.jit.load"
    except Exception as exc:
        jit_error = exc

    if loaded is None:
        try:
            loaded = torch.load(str(model_path), map_location="cpu")
            load_mode = "torch.load"
        except Exception as exc:
            print(f"torch.jit.load failed: {jit_error!r}")
            print(f"torch.load failed: {exc!r}")
            return

    print(f"load_mode: {load_mode}")
    print(f"python_type: {type(loaded)}")

    if hasattr(loaded, "state_dict"):
        try:
            sd = loaded.state_dict()
            print_state_dict(sd)
        except Exception as exc:
            print(f"state_dict() failed: {exc!r}")
    elif isinstance(loaded, dict):
        tensor_items = {k: v for k, v in loaded.items() if hasattr(v, "shape") and hasattr(v, "dtype")}
        other_keys = [k for k in loaded.keys() if k not in tensor_items]
        if tensor_items:
            print_state_dict(tensor_items)
        if other_keys:
            print("non_tensor_keys:")
            for key in other_keys:
                print(f"  {key}: {type(loaded[key])}")
    else:
        print("No state_dict available; raw object repr:")
        print(loaded)

    if hasattr(loaded, "named_modules"):
        print("== Modules ==")
        for name, module in loaded.named_modules():
            cls = module.__class__.__name__
            if name == "":
                print(f"<root>: {cls}")
            else:
                print(f"{name}: {cls}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the Kyber .pt model archive and, when torch is available, print module/state_dict tensor shapes."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to .pt model file (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sample-json",
        type=Path,
        default=DEFAULT_SAMPLE,
        help=f"Optional sample metadata json (default: {DEFAULT_SAMPLE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.resolve()

    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    inspect_archive(model_path)
    if args.sample_json:
        inspect_sample_json(args.sample_json.resolve())
    inspect_with_torch(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
