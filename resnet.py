"""Configurable 2D ResNet implementation, based on PyTorch's ResNet"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import re

import torch
import torch.nn as nn
from torch import Tensor
import torchvision


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, p: float = 0.2
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, p: float = 0.2) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        activation_function: nn.Module = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = activation_function(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        activation_function: nn.Module = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_function(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        planes: List[int],
        n_input_channels: int = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        activation_function: nn.Module = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # do_avg_pool: bool = True,
        linear_factor: Optional[int] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.activation_function = activation_function
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            n_input_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.activation = activation_function(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.do_avg_pool = linear_factor is None

        self.layers = nn.ModuleList(
            self._make_layer(block, n_planes, n_layers, stride=stride, dilate=rswd)
            for n_planes, n_layers, stride, rswd in zip(
                planes,
                layers,
                [1] + [2] * (len(layers) - 1),
                [False] + replace_stride_with_dilation,
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        linear_factor = linear_factor or 1
        self.fc = nn.Linear(linear_factor, num_classes)
        self.lessen_channels = nn.Conv2d(planes[-1], 8, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # conv1x1(self.inplanes, planes * block.expansion, 1),
                # torch.nn.MaxPool2d(stride),
                norm_layer(planes * block.expansion),
            )
            # TODO: Try replacing with max-pool

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                activation_function=self.activation_function,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    activation_function=self.activation_function,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_conv(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.lessen_channels(x)

        return x

    def _forward_linear(self, x: Tensor) -> Tensor:
        if self.do_avg_pool:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._forward_conv(x)
        x = self._forward_linear(x)

        return x


DEFAULT_PLANES = [64, 128, 256, 512]

Block = Union[Type[BasicBlock], Type[Bottleneck]]

NETS: Dict[int, Tuple[Block, List[int], List[int]]] = {
    6: (BasicBlock, [1, 1], DEFAULT_PLANES[:2]),
    8: (BasicBlock, [1, 1, 1], DEFAULT_PLANES[:3]),
    # 10: (BasicBlock, [1, 1, 1, 1], DEFAULT_PLANES),
    10: (BasicBlock, [2, 2], DEFAULT_PLANES[:2]),
    18: (BasicBlock, [2, 2, 2, 2], DEFAULT_PLANES),
    34: (BasicBlock, [3, 4, 6, 3], DEFAULT_PLANES),
    50: (Bottleneck, [3, 4, 6, 3], DEFAULT_PLANES),
    101: (Bottleneck, [3, 4, 23, 3], DEFAULT_PLANES),
    152: (Bottleneck, [3, 8, 36, 3], DEFAULT_PLANES),
}

PRETRAINED: Dict[int, torchvision.models.resnet.WeightsEnum] = {
    10: torchvision.models.resnet.ResNet18_Weights,
    18: torchvision.models.resnet.ResNet18_Weights,
    34: torchvision.models.resnet.ResNet34_Weights,
    50: torchvision.models.resnet.ResNet50_Weights,
    101: torchvision.models.resnet.ResNet101_Weights,
    152: torchvision.models.resnet.ResNet152_Weights,
}


def create_resnet(
    model_depth: int,
    pretrained: bool = False,
    do_avg_pool: bool = False,
    input_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> ResNet:
    block, layers, planes = NETS[model_depth]

    if not do_avg_pool:
        if input_size is None:
            raise ValueError

        test_net = ResNet(block=block, layers=layers, planes=planes, **kwargs)
        test_net.forward = test_net._forward_conv

        import rf
        rf_size = rf.receptivefield(test_net, (1, kwargs['n_input_channels'], *input_size))
        print(f'Output size calculation: {rf_size.outputsize}')

        # linear_factor = planes[-1] * block.expansion * rf_size.outputsize.w * rf_size.outputsize.h
        linear_factor = 8 * block.expansion * rf_size.outputsize.w * rf_size.outputsize.h
    else:
        linear_factor = None

    net = ResNet(block=block, layers=layers, planes=planes, linear_factor=linear_factor, **kwargs)

    if pretrained:
        if model_depth not in PRETRAINED.keys():
            raise ValueError

        weights = torch.hub.load_state_dict_from_url(
            PRETRAINED[model_depth].DEFAULT.url
        )
        weights['conv1.weight'] = weights['conv1.weight'].sum(dim=1, keepdim=True).repeat(1, kwargs['n_input_channels'], 1, 1)
        weights.pop('fc.weight')
        weights.pop('fc.bias')

        for key in list(weights.keys()):
            if m := re.match(r'layer(\d+)(.*)', key, re.ASCII):
                i = int(m.group(1))
                weights[f'layers.{i - 1}{m.group(2)}'] = weights.pop(key)

        ik = net.load_state_dict(weights, strict=False)
        # assert len(ik.missing_keys) == 2
        # assert len(ik.unexpected_keys) == 0

    return net
