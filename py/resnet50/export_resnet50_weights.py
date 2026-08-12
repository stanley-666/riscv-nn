# SPDX-FileContributor: Person: Stanley Lee
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torchvision.models as models


def fuse_conv_bn_eval(conv, bn):
    fused = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
    return fused


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
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
torch.save(model.state_dict(), "resnet50_state_dict.pth")

fused_model = fuse_resnet_conv_bn(model)
torch.save(fused_model.state_dict(), "resnet50_fused_state_dict.pth")

for name, param in model.state_dict().items():
    print(name)
    print("shape:", tuple(param.shape))
