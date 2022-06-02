from torch import nn
import torch
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, Optional, List


__all__ = ['BranchMobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BranchMobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes1: int = 384,
        num_classes2: int = 384,
        num_classes3: int = 384,
        num_classes4: int = 96,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
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
        super(BranchMobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],

            ]
            branch_setting = [
                # t, c, n, s
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:

            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(4, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks

        # common block
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

        features1 = []
        # 1
        input_channel1 = input_channel
        for t, c, n, s in branch_setting:
            output_channel1 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features1.append(block(input_channel1, output_channel1, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel1 = output_channel1
        # building last several layers
        features1.append(ConvBNReLU(input_channel1, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch1_features = nn.Sequential(*features1)

        # building classifier
        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes1),
        )

        features2 = []
        # 2
        input_channel2 = input_channel
        for t, c, n, s in branch_setting:
            output_channel2 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features2.append(block(input_channel2, output_channel2, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel2 = output_channel2
        # building last several layers
        features2.append(ConvBNReLU(input_channel2, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch2_features = nn.Sequential(*features2)

        # building classifier
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes2),
        )

        #3
        input_channel3 = input_channel
        features3 = []
        for t, c, n, s in branch_setting:
            output_channel3 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features3.append(block(input_channel3, output_channel3, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel3 = output_channel3
        # building last several layers
        features3.append(ConvBNReLU(input_channel3, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch3_features = nn.Sequential(*features3)

        # building classifier
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes3),
        )

        #4
        input_channel4 = input_channel
        features4 = []
        for t, c, n, s in branch_setting:
            output_channel4 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features4.append(block(input_channel4, output_channel4, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel4 = output_channel4
        # building last several layers
        features4.append(ConvBNReLU(input_channel4, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch4_features = nn.Sequential(*features4)

        # building classifier
        self.classifier4 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes4),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        common = self.features(x)
        # print(common.shape)

        branch1 = self.branch1_features(common)
        # print(branch1.shape)
        branch1 = nn.functional.adaptive_avg_pool2d(branch1, (1, 1)).reshape(branch1.shape[0], -1)
        # print(branch1.shape)
        branch1 = self.classifier1(branch1)
        # print(branch1.shape)
        # print("_____________________")

        # print(common.shape)
        branch2 = self.branch2_features(common)
        # print(branch2.shape)
        branch2 = nn.functional.adaptive_avg_pool2d(branch2, (1, 1)).reshape(branch2.shape[0], -1)
        # print(branch2.shape)
        branch2 = self.classifier2(branch2)
        # print(branch2.shape)

        branch3 = self.branch3_features(common)
        branch3 = nn.functional.adaptive_avg_pool2d(branch3, (1, 1)).reshape(branch3.shape[0], -1)
        branch3 = self.classifier3(branch3)

        branch4 = self.branch4_features(common)
        branch4 = nn.functional.adaptive_avg_pool2d(branch4, (1, 1)).reshape(branch4.shape[0], -1)
        branch4 = self.classifier4(branch4)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]

        return branch1, branch2, branch3, branch4

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BranchMobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = BranchMobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# net = BranchMobileNetV2()  # resnet.resnet18()#
# inp = torch.rand((8, 4, 224, 224))
# net(inp)
# print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))