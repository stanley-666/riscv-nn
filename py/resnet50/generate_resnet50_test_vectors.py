# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
import argparse
import time
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


def write_text_array(path, data, cols=8):
    with open(path, "w", encoding="ascii") as f:
        for i, val in enumerate(data):
            if i % cols == 0:
                f.write("  ")
            f.write(f"{float(val):.9f}")
            if i != len(data) - 1:
                f.write(", ")
            if i % cols == cols - 1:
                f.write("\n")
        if len(data) % cols != 0:
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Generate ResNet50 test vectors.")
    parser.add_argument("--weights", default="resnet50_fused_state_dict.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-input", default="resnet50_input.txt")
    parser.add_argument("--out-output", default="resnet50_output.txt")
    parser.add_argument("--bin", action="store_true", help="Write float32 .bin files instead of text.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    input_data = np.random.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)
    input_tensor = torch.from_numpy(input_data)

    model = models.resnet50(weights=None)
    model = fuse_resnet_conv_bn(model)
    state_dict = torch.load(args.weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("missing_keys:", missing)
        print("unexpected_keys:", unexpected)

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        output_tensor = model(input_tensor)
        end = time.perf_counter()
        print(f"cpu inference time: {(end - start) * 1000:.3f} ms")

    in_flat = input_tensor.detach().cpu().contiguous().view(-1).numpy()
    out_flat = output_tensor.detach().cpu().contiguous().view(-1).numpy()

    if args.bin:
        in_flat.astype(np.float32).tofile(args.out_input)
        out_flat.astype(np.float32).tofile(args.out_output)
    else:
        write_text_array(args.out_input, in_flat)
        write_text_array(args.out_output, out_flat)

    print(f"wrote input: {args.out_input} ({in_flat.size} floats)")
    print(f"wrote output: {args.out_output} ({out_flat.size} floats)")


if __name__ == "__main__":
    main()
