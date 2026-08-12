# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
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


def write_header(state_dict, header_path):
    with open(header_path, "w", encoding="ascii") as f:
        f.write("#ifndef RESNET50_WEIGHTS_H\n")
        f.write("#define RESNET50_WEIGHTS_H\n\n")
        f.write("#include <stddef.h>\n\n")

        for name, tensor in state_dict.items():
            array_name = name.replace(".", "_")
            shape = tuple(tensor.shape)
            if tensor.dim() == 4:
                f.write(f"// {name} shape (oc, ic, h, w): {shape}\n")
                f.write("// flatten order: oc -> ic -> h -> w (w is fastest)\n")
            elif tensor.dim() == 2:
                f.write(f"// {name} shape (out, in): {shape}\n")
                f.write("// flatten order: out -> in (in is fastest)\n")
            else:
                f.write(f"// {name} shape: {shape}\n")
            f.write(f"const float {array_name}[{tensor.numel()}] = {{\n")

            data = tensor.detach().cpu().contiguous().view(-1).numpy()
            for i, val in enumerate(data):
                if i % 8 == 0:
                    f.write("  ")
                f.write(f"{float(val):.9f}")
                if i != len(data) - 1:
                    f.write(", ")
                if i % 8 == 7:
                    f.write("\n")
            if len(data) % 8 != 0:
                f.write("\n")
            f.write("};\n\n")

        f.write("#endif  // RESNET50_WEIGHTS_H\n")


def write_io_header(input_tensor, output_tensor, header_path):
    with open(header_path, "w", encoding="ascii") as f:
        f.write("#ifndef RESNET50_IO_H\n")
        f.write("#define RESNET50_IO_H\n\n")
        f.write("#include <stddef.h>\n\n")

        in_shape = tuple(input_tensor.shape)
        out_shape = tuple(output_tensor.shape)
        f.write(f"// input shape (n, c, h, w): {in_shape}\n")
        f.write("// input flatten order: n -> c -> h -> w (w is fastest)\n")
        f.write(
            "#define RESNET50_INPUT_INDEX(n, c, h, w, C, H, W) "
            "((((n) * (C) + (c)) * (H) + (h)) * (W) + (w))\n"
        )
        f.write(f"const float resnet50_input[{input_tensor.numel()}] = {{\n")
        in_data = input_tensor.detach().cpu().contiguous().view(-1).numpy()
        for i, val in enumerate(in_data):
            if i % 8 == 0:
                f.write("  ")
            f.write(f"{float(val):.9f}f")
            if i != len(in_data) - 1:
                f.write(", ")
            if i % 8 == 7:
                f.write("\n")
        if len(in_data) % 8 != 0:
            f.write("\n")
        f.write("};\n\n")

        f.write(f"// output shape: {out_shape}\n")
        f.write("// output flatten order: n -> classes (classes is fastest)\n")
        f.write(
            "#define RESNET50_OUTPUT_INDEX(n, k, K) (((n) * (K)) + (k))\n"
        )
        f.write(f"const float resnet50_output[{output_tensor.numel()}] = {{\n")
        out_data = output_tensor.detach().cpu().contiguous().view(-1).numpy()
        for i, val in enumerate(out_data):
            if i % 8 == 0:
                f.write("  ")
            f.write(f"{float(val):.9f}f")
            if i != len(out_data) - 1:
                f.write(", ")
            if i % 8 == 7:
                f.write("\n")
        if len(out_data) % 8 != 0:
            f.write("\n")
        f.write("};\n\n")

        f.write("#endif  // RESNET50_IO_H\n")


model = models.resnet50(weights=None)
model = fuse_resnet_conv_bn(model)
state_dict = torch.load("resnet50_fused_state_dict.pth", map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("missing_keys:", missing)
print("unexpected_keys:", unexpected)

for module_name, module in model.named_modules():
    params = list(module.named_parameters(recurse=False))
    if not params:
        continue
    print(module_name if module_name else "<root>")
    for param_name, param in params:
        print(f"  {param_name} shape: {tuple(param.shape)}")

write_header(state_dict, "resnet50_weights.h")
print("wrote resnet50_weights.h")

torch.manual_seed(0)
np.random.seed(0)
input_data = np.random.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)
input_tensor = torch.from_numpy(input_data)
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)

write_io_header(input_tensor, output_tensor, "resnet50_io.h")
print("wrote resnet50_io.h")
