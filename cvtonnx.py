#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:04:22 2020

@author: apline
"""
import torch
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.onnx as torch_onnx

model = models.resnet50()
# model = models.shufflenet_v2_x1_0()
# model.classifier[1] = nn.Linear(1280,220)
# model.fc=nn.Linear(1024,220)
# model = models.mobilenet_v2()
# from moblienet_new import mobilenet_v2
import ownTiny0615
from net0613 import mbv2_ca0613

# model = mobilenet_v2()
# model.classifier[1] = nn.Linear(1280, 1248)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model = models.shufflenet_v2_x2_0()
# model.fc=nn.Linear(2048,1248)
# model.conv1[0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model = ownTiny0615.mobilenet_v2()
# model = models.resnet50()
# model.fc=nn.Linear(2048,1248)
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
print(model)
# model.fc=nn.Linear(4096,1248)
# model.features.conv0 = nn.Conv2d(4, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc =nn.Linear(2048,220)
# model.Conv2d_1a_3x3.conv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.conv1[0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model = torch.nn.DataParallel(model)
# checkpoint = torch.load("/home/kai/Desktop/AlpineProject/tinyown0615_2211.pth.tar")
# model.load_state_dict(checkpoint)
input_shape = (3, 224, 224)
model_onnx_path = "resnet50_random.onnx"
# if isinstance(model, torch.nn.DataParallel):
        # model = model.module
model.train(False)
model.cuda()

# Export the model to an ONNX file
dummy_input = Variable(torch.randn(1, *input_shape).cuda())
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False, opset_version=11)
print("Export of torch_model.onnx complete!")