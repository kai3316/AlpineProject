import torch
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2

class mobilenet_branch_network(nn.Module):
    def __init__(self, branch1_classes=384, branch2_classes=384, branch3_classes=384, branch4_classes=96, channel=2048):
        super(mobilenet_branch_network, self).__init__()
        # This is the refernce net which we will decompose
        rn = mobilenet_v2()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cb = nn.ModuleList([rn.layer1, rn.layer2, rn.layer3])
        self.mbf1 = rn.layer4  # 0m-1m risk detector
        self.mbl1 = nn.Linear(channel, branch1_classes)  # 0m-1m risk detector
        self.mbf2 = rn.layer4  # 1m-2m risk detector
        self.mbl2 = nn.Linear(channel, branch2_classes)  # 1m-2m risk detector
        self.mbf3 = rn.layer4  # 2m-6m risk detector
        self.mbl3 = nn.Linear(channel, branch3_classes)  # 2m-6m risk detector
        self.mbf4 = rn.layer4  # 2m-6m risk detector
        self.mbl4 = nn.Linear(channel, branch4_classes)  # 2m-6m risk detector
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print (reference_net.layer1)

    def forward(self, x):
        common_out = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        # common features from common block (cb)
        for lay in self.cb:
            common_out = lay(common_out)

        branch1_out = self.mbf1(common_out)
        branch1_out = self.avgpool(branch1_out)
        branch1_out = torch.flatten(branch1_out, 1)
        branch1_out = self.mbl1(branch1_out)

        branch2_out = self.mbf2(common_out)
        branch2_out = self.avgpool(branch2_out)
        branch2_out = torch.flatten(branch2_out, 1)
        branch2_out = self.mbl2(branch2_out)

        branch3_out = self.mbf3(common_out)
        branch3_out = self.avgpool(branch3_out)
        branch3_out = torch.flatten(branch3_out, 1)
        branch3_out = self.mbl3(branch3_out)

        branch4_out = self.mbf4(common_out)
        branch4_out = self.avgpool(branch4_out)
        branch4_out = torch.flatten(branch4_out, 1)
        branch4_out = self.mbl4(branch4_out)

        return branch1_out, branch2_out, branch3_out, branch4_out

        # if (sg == 'sub_0'):
        #     out1 = self.hbf1(out)
        #     out1f = F.avg_pool2d(out1, out1.size()[3])
        #     out1f = out1f.view(out1f.size(0), -1)
        #     out1f = self.hbl1(out1f)
        #     return out1f
        # if (sg == 'sub_1'):
        #     out2 = self.hbf2(out)
        #     out2f = F.avg_pool2d(out2, out2.size()[3])
        #     out2f = out2f.view(out2f.size(0), -1)
        #     out2f = self.hbl2(out2f)
        #     # out = F.log_softmax(out)
        #     return out2f
        # if (sg == 'sub_2'):
        #     out3 = self.hbf3(out)
        #     out3f = F.avg_pool2d(out3, out3.size()[3])
        #     out3f = out3f.view(out3f.size(0), -1)
        #     out3f = self.hbl3(out3f)
        #     # out = F.log_softmax(out)
        #     return out3f
        # out1 = self.hbf1(out)  # you can do better implementation by saving them in module list.
        # # here I save them in several variable. (I am not a good coder!!)
        # out2 = self.hbf2(out)
        # out3 = self.hbf3(out)  # From the router/common block
        # outcat = torch.cat((out1, out2, out3), 1)
        # outcat = F.avg_pool2d(outcat, outcat.size()[3])
        # outcat = outcat.view(out.size(0), -1)
        # outfinal = self.finale(outcat)
        # return outfinal


# net = branch_network()  # resnet.resnet18()#
# inp = torch.rand((8, 3, 224, 224))
# # print (net(inp))
# print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))