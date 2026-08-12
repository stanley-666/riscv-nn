#!/usr/bin/env python3
# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
"""
Export FP32 weights/bias from SentenceCNN checkpoint to a C header.

Usage (run from repo root):
  python tools/export_fp32_header.py \
      --ckpt py/sentence/weights/best_weights.pth \
      --out testbench/sentence_inference_fp32_test/weights_fp32.h
"""

import argparse
import pathlib
import torch
import numpy as np
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "py" / "sentence"))
from sentence_infer import SentenceCNN  # noqa: E402


def write_array(f, name, arr: np.ndarray, ctype="float"):
    flat = arr.astype(np.float32).ravel()
    f.write(f"const {ctype} {name}[{flat.size}] = " "{")
    f.write(",".join(f"{v:.9g}" for v in flat))
    f.write("};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="PyTorch checkpoint (.pth)")
    ap.add_argument("--out", required=True, help="Output header path")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = SentenceCNN()
    model.load_state_dict(state, strict=False)
    model.eval()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("#ifndef WEIGHTS_FP32_H\n#define WEIGHTS_FP32_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Conv layers: OIK flattened
        for name in ["conv1", "conv2", "conv3"]:
            w = dict(model.named_parameters())[f"{name}.weight"].detach().cpu().numpy()  # (O,I,K)
            b = dict(model.named_parameters())[f"{name}.bias"].detach().cpu().numpy()
            write_array(f, f"{name}_weight", w)
            write_array(f, f"{name}_bias", b)

        # FC layers: (O,I) flattened
        for name in ["fc1", "fc2"]:
            w = dict(model.named_parameters())[f"{name}.weight"].detach().cpu().numpy()  # (O,I)
            b = dict(model.named_parameters())[f"{name}.bias"].detach().cpu().numpy()
            write_array(f, f"{name}_weight", w)
            write_array(f, f"{name}_bias", b)

        f.write("#endif // WEIGHTS_FP32_H\n")

    print(f"[export] wrote {out_path}")


if __name__ == "__main__":
    main()
