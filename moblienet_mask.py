import math

import torch
import warnings

from functools import partial
from torch import nn
from torch import Tensor

from typing import Callable, Any, Optional, List

from torch.hub import load_state_dict_from_url


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 7,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.GroupNorm,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.GELU,
            dilation: int = 1,
            inplace: bool = True,
            groups_norm: int = 16,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(groups_norm, out_channels))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# necessary for backwards compatibility
class _DeprecatedConvBNAct(ConvNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated and will be removed in future versions. "
            "Use torchvision.ops.misc.ConvNormActivation instead.", FutureWarning)
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.GroupNorm
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.GELU
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None, keep_3x3=False, dwKernel_size=7, num_channels=2):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))


        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        # dw
        if expand_ratio == 2 or inp == oup or keep_3x3:
            layers.append(ConvBNReLU(inp, inp, kernel_size=dwKernel_size,padding=dwKernel_size//2, stride=1, groups=inp, norm_layer=norm_layer, activation_layer=None))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                # norm_layer(num_channels, hidden_dim),
            ])
        layers.extend([CoordAtt(hidden_dim, hidden_dim)])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
        ])
        if expand_ratio == 2 or inp == oup or keep_3x3 or stride == 2:
            layers.extend([
                # dw-linear
                nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
                # norm_layer(num_channels,oup),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] + out[:, :self.identity_tensor_channels, :,
                                                                               :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 1,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.GroupNorm

        input_channel = 96
        last_channel = 12

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, k
                [2, 128, 2, 1, 7],
                [4, 256, 3, 2, 7],
                [4, 512, 4, 2, 7],
                [4, 1024, 2, 1, 7],
            ]
        # [2, 96, 1, 2],
        # [6, 144, 1, 1],
        # [6, 192, 3, 2],
        # [6, 288, 3, 2],
        # [6, 384, 4, 1],
        # [6, 576, 4, 2],
        # [6, 960, 2, 1],
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        stem: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=4, norm_layer=norm_layer, kernel_size=4,
                                                    activation_layer=nn.GELU)]
        # self.stem = nn.ModuleList([
        #     ConvNormActivation(in_channels=input_channel, out_channels=3, kernel_size=3, stride=2, padding=1, groups=1),
        #     ConvNormActivation(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
        #     ConvNormActivation(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=1),
        #     ConvNormActivation(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, groups=3)])
        self.stem = nn.Sequential(*stem)

        block1: List[nn.Module] = []
        t, c, n, s, k = inverted_residual_setting[0]
        # building inverted residual blocks
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            block1.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, dwKernel_size=k))
            input_channel = output_channel
        # make it nn.Sequential
        self.block1 = nn.Sequential(*block1)

        block2: List[nn.Module] = []
        t, c, n, s, k = inverted_residual_setting[1]
        # building inverted residual blocks
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            block2.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, dwKernel_size=k))
            input_channel = output_channel
        # make it nn.Sequential
        self.block2 = nn.Sequential(*block2)

        block3: List[nn.Module] = []
        t, c, n, s, k = inverted_residual_setting[2]
        # building inverted residual blocks
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            block3.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, dwKernel_size=k))
            input_channel = output_channel
        # make it nn.Sequential
        self.block3 = nn.Sequential(*block3)

        block4: List[nn.Module] = []
        t, c, n, s, k = inverted_residual_setting[3]
        # building inverted residual blocks
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            block4.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, dwKernel_size=k))
            input_channel = output_channel
        # make it nn.Sequential
        self.block4 = nn.Sequential(*block4)

        last: List[nn.Module] = []
        # building last several layers
        # print(input_channel)
        last.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=None,
                                       activation_layer=None))
        # # make it nn.Sequential
        self.last = nn.Sequential(*last)
        #
        # # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.stem(x)
        # print(x.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #resize x to the original size
        x = torch.nn.functional.interpolate(x, (58,58), mode='bilinear', align_corners=True)
        # print(x.shape)
        x = self.last(x)
        # print(x.shape)
        # Cannot use "squeeze" as batch-size can be 1
        # print(x.shape)
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    import torchvision.models as models
    net = mobilenet_v2()  # resnet.resnet18()#
    # net  = models.mobilenet_v2()
    print(net)
    inp = torch.rand((8, 3, 224, 224))
    out = net(inp)
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
