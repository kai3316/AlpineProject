from torch import nn
import torch
from torch import Tensor

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, Optional, List
from aspp import ASPP, ASPP_Bottleneck
import os

__all__ = ['YOLIC_MobileNetV2']

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
            padding: int = 0,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
    ) -> None:
        if stride != 1:
            padding = dilation
        else:
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


class ConvBNActivation(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            padding: int = 0,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
    ) -> None:
        if stride != 1:
            padding = dilation
        else:
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


class deConvBNActivation(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            padding: int = 0,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
    ) -> None:
        if stride != 1:
            padding = dilation
        else:
            padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(deConvBNReLU, self).__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation
deConvBNReLU = deConvBNActivation

class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            dilation: int,
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
        padding = 0
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            if stride == 2:
                dilation = 2
                padding = dilation
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim, padding=padding,
                       norm_layer=norm_layer),
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


class YOLIC_MobileNetV2(nn.Module):
    def __init__(
            self,
            outputChannel: int = 20,
            num_classes: int = 20,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            project_dir: str = "",
            model_id: str = "1"
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
        super(YOLIC_MobileNetV2, self).__init__()
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()
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
                [6, 32, 3, 1],
                [6, 64, 4, 1],

            ]
            branch2_setting = [
                # t, c, n, s
                [6, 96, 3, 2],
                [6, 160, 3, 2],
                [6, 320, 1, 2],
            ]
            branch5_setting = [
                # t, c, n, s
                [6, 96, 3, 2],
                [6, 160, 3, 2],
                [6, 320, 1, 2],
            ]
            branch3_setting = [
                # t, c, n, s
                [6, 96, 3, 1],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks

        # common block
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, dilation=1, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

        self.branch1_out = ConvBNReLU(64, outputChannel, kernel_size=1, norm_layer=norm_layer)

        features1 = []
        # 1
        input_channel1 = input_channel
        for t, c, n, s in branch2_setting:
            output_channel1 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features1.append(
                    block(input_channel1, output_channel1, stride, dilation=1, expand_ratio=t, norm_layer=norm_layer))
                input_channel1 = output_channel1
        # building last several layers
        features1.append(ConvBNReLU(input_channel1, 512, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch1_features = nn.Sequential(*features1)
        self.branch2_out = ConvBNReLU(512, outputChannel, kernel_size=1, norm_layer=norm_layer)
        self.deconv1 = deConvBNReLU(64, 64, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.deconv2 = deConvBNReLU(64, 64, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.deconv3 = deConvBNReLU(64, 64, kernel_size=3, stride=2, norm_layer=norm_layer)
        # building classifier
        # self.classifier1 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes1),
        # )

        features2 = []
        # 2
        input_channel2 = input_channel
        for t, c, n, s in branch3_setting:
            output_channel2 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features2.append(
                    block(input_channel2, output_channel2, stride, dilation=1, expand_ratio=t, norm_layer=norm_layer))
                input_channel2 = output_channel2
        # building last several layers
        features2.append(ConvBNReLU(input_channel2, 512, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch2_features = nn.Sequential(*features2)
        self.aspp = ASPP(
            num_classes=num_classes)  # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        features5 = []
        # 2
        input_channel5 = input_channel
        for t, c, n, s in branch5_setting:
            output_channel5 = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features5.append(
                    block(input_channel5, output_channel5, stride, dilation=1, expand_ratio=t, norm_layer=norm_layer))
                input_channel5 = output_channel5
        # building last several layers
        features5.append(ConvBNReLU(input_channel5, 64, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.branch5_features = nn.Sequential(*features5)
        # # building classifier
        # self.classifier2 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes2),
        # )

        # 3
        # input_channel3 = input_channel
        # features3 = []
        # for t, c, n, s in branch3_setting:
        #     output_channel3 = _make_divisible(c * width_mult, round_nearest)
        #     for i in range(n):
        #         stride = s if i == 0 else 1
        #         features3.append(block(input_channel3, output_channel3, stride, dilation=1, expand_ratio=t, norm_layer=norm_layer))
        #         input_channel3 = output_channel3
        # # building last several layers
        # features3.append(ConvBNReLU(input_channel3, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # # make it nn.Sequential
        # self.branch3_features = nn.Sequential(*features3)
        #
        # # building classifier
        # self.classifier3 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes3),
        # )

        # # 4
        # input_channel4 = input_channel
        # features4 = []
        # for t, c, n, s in branch4_setting:
        #     output_channel4 = _make_divisible(c * width_mult, round_nearest)
        #     for i in range(n):
        #         stride = s if i == 0 else 1
        #         features4.append(block(input_channel4, output_channel4, stride, expand_ratio=t, norm_layer=norm_layer))
        #         input_channel4 = output_channel4
        # # building last several layers
        # features4.append(ConvBNReLU(input_channel4, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # # make it nn.Sequential
        # self.branch4_features = nn.Sequential(*features4)
        #
        # # building classifier
        # self.classifier4 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes4),
        # )

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

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "./training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        self.results_dir = self.model_dir + "/results"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _forward_impl(self, x, mode, branchSetting, commonTensor):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        if mode == 'branch1':
            common = self.features(x)  # torch.Size([2, 64, 94, 311])
            common1 = self.branch5_features(common)
            # print("common:", common1.shape)
            common = self.deconv1(common)
            common = self.deconv2(common)
            # common = self.deconv3(common)
            # print(common.shape)
            # exit(0)
            out_feature = nn.functional.interpolate(common, size=(375, 1242), mode='nearest')
            common_feature = nn.functional.interpolate(common1, size=(5, 6), mode='nearest')
            branch1_output = self.branch1_out(common_feature)
            return branch1_output, out_feature

        if mode == 'branch2':
            """Feature_branch1: (2,64, 375, 1242), step size: (75, 207)"""
            point_x = branchSetting["x"] * 75
            point_y = branchSetting["y"] * 207
            # print(x.shape)
            # print(point_x, point_y)
            interesting_region = x[:, :, point_x: point_x + 75, point_y: point_y + 207]

            # interesting_region = nn.functional.interpolate(interesting_region, size=(37, 103), mode='bilinear', align_corners=True)
            # print(interesting_region.shape)
            # interesting_region = x
            feature_map = self.branch1_features(interesting_region)
            # print(feature_map.shape)
            # feature_map = self.aspp(feature_map)
            feature_map = nn.functional.interpolate(feature_map, size=(5, 9), mode='nearest')
            feature_map = self.branch2_out(feature_map)

            return feature_map

        if mode == 'branch3':
            point_x = branchSetting["x"] * 15
            point_y = branchSetting["y"] * 23
            # print(x.shape)
            # print(point_x, point_y)
            interesting_region = x[:, :, point_x: point_x + 15, point_y: point_y + 23]
            # print(interesting_region.shape) # (2, 64, 15, 23)
            # out_feature = nn.functional.interpolate(interesting_region, size=(37, 103), mode='bilinear',
            #                                         align_corners=False)
            # print(out_feature.shape)
            feature_map = self.branch2_features(interesting_region)
            # print(feature_map.shape)
            output = self.aspp(feature_map)
            # output = nn.functional.interpolate(output, size=(75, 207), mode='bilinear', align_corners=False)

            return output

    def forward(self, x: Tensor, mode, branchSetting=None, commonTensor=None):
        return self._forward_impl(x, mode, branchSetting, commonTensor)


import numpy as np

if __name__ == '__main__':
    net = YOLIC_MobileNetV2()  # resnet.resnet18()#
    inp = torch.rand((2, 3, 375, 1242))
    commonOut, commonFeature = net(inp, "branch1")
    # print(commonOut.shape , commonFeature.shape)
    # branch_out = net(inp, "branch2", {"x": 0, "y": 0})
    # branch3Setting = {"x": np.random.randint(0, 25), "y": np.random.randint(0, 54)}
    # print(branch3Setting["x"], branch3Setting["y"])
    # branch_out = net(inp, "branch3", branch3Setting)
# print(branch_out.shape)
# print(out)
# print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
