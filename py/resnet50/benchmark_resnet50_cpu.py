import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def fuse_conv_bn_eval(conv, bn):
    return torch.nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_resnet_conv_bn(model):
    model.eval()

    model.conv1 = fuse_conv_bn_eval(model.conv1, model.bn1)
    model.bn1 = nn.Identity()

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = nn.Identity()

            block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = nn.Identity()

            if hasattr(block, "conv3") and hasattr(block, "bn3"):
                block.conv3 = fuse_conv_bn_eval(block.conv3, block.bn3)
                block.bn3 = nn.Identity()

            if block.downsample is not None:
                if isinstance(block.downsample, nn.Sequential) and len(block.downsample) >= 2:
                    block.downsample[0] = fuse_conv_bn_eval(block.downsample[0], block.downsample[1])
                    block.downsample[1] = nn.Identity()

    return model


def load_input(input_path: Path) -> torch.Tensor:
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    if input_path.suffix == ".bin":
        flat = np.fromfile(input_path, dtype=np.float32)
    else:
        flat = np.loadtxt(input_path, delimiter=",", dtype=np.float32)

    expected = 1 * 3 * 224 * 224
    if flat.size != expected:
        raise ValueError(f"expected {expected} floats, got {flat.size}")

    return torch.from_numpy(flat.reshape(1, 3, 224, 224))


def main():
    parser = argparse.ArgumentParser(description="Benchmark single-sample ResNet50 inference on CPU.")
    parser.add_argument("--weights", default="resnet50_fused_state_dict.pth")
    parser.add_argument("--input", default="resnet50_input.txt")
    parser.add_argument("--runs", type=int, default=1, help="Repeat inference to average runtime.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations not timed.")
    args = parser.parse_args()

    device = torch.device("cpu")

    input_tensor = load_input(Path(args.input)).to(device)

    model = models.resnet50(weights=None)
    model = fuse_resnet_conv_bn(model)
    state_dict = torch.load(args.weights, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("warning: mismatched keys", missing, unexpected)

    model.eval()
    model.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_tensor)

        start = time.perf_counter()
        for _ in range(args.runs):
            _ = model(input_tensor)
        end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / max(args.runs, 1)
    print(f"CPU inference avg over {args.runs} run(s): {avg_ms:.3f} ms")


if __name__ == "__main__":
    main()
