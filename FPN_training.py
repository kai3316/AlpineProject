#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:39:10 2020

@author: Su
"""

from PIL import Image
import argparse
import numpy as np
import torch
import torchvision
from torchvision import models
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.nn as nn
import time
import copy
import cv2
import os.path
import matplotlib.pyplot as plt
import os
from FPNnet import MobileNetV2_dynamicFPN

torch.multiprocessing.freeze_support()
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')

torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    transforms.ToTensor()
]))

rgb_dir = r'C:\Users\Kai\Desktop\AlpineData\RGB'
depth_dir = r'C:\Users\Kai\Desktop\AlpineData\Depth'
label_dir = r'C:\Users\Kai\Desktop\AlpineData\yoliclabel'
# rgbdir = '/media/apline/2TSSD/34294/RGB'
# depthdir = '/media/apline/2TSSD/34294/Depth'
# labeldir = '/media/apline/2TSSD/34294/yoliclabel'

img_list = os.listdir(rgb_dir)
x_train, x_test = train_test_split(img_list, test_size=0.3, random_state=2)
train = MultiLabelDataSet(rgb_dir, depth_dir, x_train, label_dir, trans, 1)
test = MultiLabelDataSet(rgb_dir, depth_dir, x_test, label_dir, trans, 0)
# train = RealsenseDataSet(x_train,val_trans)
# test = RealsenseDataSet(x_test,val_trans)
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=11)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=11)

model = MobileNetV2_dynamicFPN()
# model.fc = nn.Linear(2048, 1248)
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # model = models.alexnet()

# model = models.mobilenet_v2()
# model.classifier[1] = nn.Linear(1280,1248)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


# model = models.shufflenet_v2_x2_0()
# model.fc=nn.Linear(2048,1248)
# model.conv1[0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

# from mixnet import MixNet
# model = MixNet(net_type='mixnet_s', input_size=224, num_classes=1248)
# model.stem_conv[0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.classifier[1] = nn.Linear(1280,1248)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.fc=nn.Linear(4096,1248)
# model = models.vgg16()
# model.classifier[6] = nn.Linear(4096,22)
# print(model)
# teacher = VGG16()

model = torch.nn.DataParallel(model)
save_name = 'FPN_1248_5_40760'
checkpoint = torch.load(os.path.join(os.getcwd(), "FPN_1248_5_40760.pth.tar"))
model.load_state_dict(checkpoint)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
criterion = nn.MultiLabelSoftMarginLoss()
best_correct = -999
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler = StepLR(optimizer,step_size = 50,gamma=0.1)


def pred_acc(original, predicted, All_Two):
    pred = torch.round(predicted).detach().numpy()
    orig = original.detach().numpy()
    num = 0
    enum = 0
    normal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    for cell in range(0, 1248, 12):
        if (orig[cell:cell + 12] == pred[cell:cell + 12]).all():
            num = num + 1
        else:
            if not (orig[cell:cell + 12] == normal).all() and not (pred[cell:cell + 12] == normal).all():
                enum = enum + 1
    if All_Two == -1:
        return (num + enum) / 104
    if All_Two == 0:
        return num / 104


def train(epoch, model, loss_fn):
    model.train()
    correct = 0
    # i =0
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        outputs = model(data)
        loss = 0
        for output in outputs:
            loss = loss + loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def train_evaluate(model):
    model.eval()
    global best_correct
    running_loss = [[] for i in range(5)]
    AllClass_acc = [[] for i in range(5)]
    TwoClass_acc = [[] for i in range(5)]
    total_batch_acc = [[] for i in range(5)]
    total_batch_acc_two = [[] for i in range(5)]
    total_batch_loss = [[] for i in range(5)]
    running_acc_avg = []
    running_accTwo_avg = []
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        outputs = model(data)
        loss = [[] for i in range(5)]
        acc = [[] for i in range(5)]
        acc_two = [[] for i in range(5)]
        avg = torch.zeros_like(outputs[0]).cuda()
        for output in outputs:
            output = torch.sigmoid(output)
            avg = avg + output
        avg = avg / 5
        acc_avg = []
        accTwo_avg = []
        for each_image, d in enumerate(avg):
            acc_avg.append(pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), 0))
            accTwo_avg.append(pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), -1))
        running_acc_avg.append(np.asarray(acc_avg).mean())
        running_accTwo_avg.append(np.asarray(accTwo_avg).mean())

        for num, output in enumerate(outputs):
            loss_out = criterion(output, target)
            loss[num].append(loss_out.item())
            output = torch.sigmoid(output)
            for i, d in enumerate(output):
                acc[num].append(pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0))
                acc_two[num].append(pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), -1))
        for i in range(5):
            AllClass_acc[i].append(np.asarray(acc[i]).mean())
            TwoClass_acc[i].append(np.asarray(acc_two[i]).mean())
            running_loss[i].append(np.asarray(loss[i]).mean())

    for i in range(5):
        total_batch_acc[i] = np.asarray(AllClass_acc[i]).mean()
        total_batch_acc_two[i] = np.asarray(TwoClass_acc[i]).mean()
        total_batch_loss[i] = np.asarray(running_loss[i]).mean()
        print('Train set from {}: total_batch_loss: {:.4f}, total images: {} , AllClass_accuracy: ({:.4f}%), '
              'TwoClass_accuracy: ({:.4f}%)'.format(i, total_batch_loss[i], len(train_loader.dataset),
                                                    total_batch_acc[i], total_batch_acc_two[i]))
    total_batch_acc_avg = np.asarray(running_acc_avg).mean()
    total_batch_accTwo_avg = np.asarray(running_accTwo_avg).mean()
    print('Train set from All: total_batch_loss: {:.4f}, total images: {} , AllClass_accuracy: ({:.4f}%), '
          'TwoClass_accuracy: ({:.4f}%)\n'.format(total_batch_loss[i], len(train_loader.dataset), total_batch_acc_avg,
                                                  total_batch_accTwo_avg))
    return total_batch_acc_avg, total_batch_accTwo_avg


def test(model):
    model.eval()
    global best_correct
    running_loss = [[] for i in range(5)]
    AllClass_acc = [[] for i in range(5)]
    TwoClass_acc = [[] for i in range(5)]
    total_batch_acc = [[] for i in range(5)]
    total_batch_acc_two = [[] for i in range(5)]
    total_batch_loss = [[] for i in range(5)]
    running_acc_avg = []
    running_accTwo_avg = []
    for batch_idx, (data, target, filenames) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        outputs = model(data)
        loss = [[] for i in range(5)]
        acc = [[] for i in range(5)]
        acc_two = [[] for i in range(5)]
        avg = torch.zeros_like(outputs[0]).cuda()
        for output in outputs:
            output = torch.sigmoid(output)
            avg = avg + output
        avg = avg / 5
        acc_avg = []
        accTwo_avg = []
        for each_image, d in enumerate(avg):
            acc_avg.append(pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), 0))
            accTwo_avg.append(pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), -1))
        running_acc_avg.append(np.asarray(acc_avg).mean())
        running_accTwo_avg.append(np.asarray(accTwo_avg).mean())

        for num, output in enumerate(outputs):
            loss_out = criterion(output, target)
            loss[num].append(loss_out.item())
            output = torch.sigmoid(output)
            for i, d in enumerate(output):
                acc[num].append(pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0))
                acc_two[num].append(pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), -1))
        for i in range(5):
            AllClass_acc[i].append(np.asarray(acc[i]).mean())
            TwoClass_acc[i].append(np.asarray(acc_two[i]).mean())
            running_loss[i].append(np.asarray(loss[i]).mean())

    for i in range(5):
        total_batch_acc[i] = np.asarray(AllClass_acc[i]).mean()
        total_batch_acc_two[i] = np.asarray(TwoClass_acc[i]).mean()
        total_batch_loss[i] = np.asarray(running_loss[i]).mean()
        print('Train set from {}: total_batch_loss: {:.4f}, total images: {} , AllClass_accuracy: ({:.4f}%), '
              'TwoClass_accuracy: ({:.4f}%)'.format(i, total_batch_loss[i], len(train_loader.dataset),
                                                    total_batch_acc[i], total_batch_acc_two[i]))
    total_batch_acc_avg = np.asarray(running_acc_avg).mean()
    total_batch_accTwo_avg = np.asarray(running_accTwo_avg).mean()
    print('Train set from All: total_batch_loss: {:.4f}, total images: {} , AllClass_accuracy: ({:.4f}%), '
          'TwoClass_accuracy: ({:.4f}%)\n'.format(total_batch_loss[i], len(train_loader.dataset), total_batch_acc_avg,
                                                  total_batch_accTwo_avg))
    if best_correct < total_batch_acc_avg:
        print("New weight!")
        best_correct = total_batch_acc_avg
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(os.getcwd(), save_name))
    return total_batch_acc_avg, total_batch_accTwo_avg


class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, dpath, imgslist, annotationpath, transforms=None, mode=0):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.dpath = dpath
        self.transform = transforms
        self.annotationpath = annotationpath
        self.mode = mode
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        dpath = os.path.join(self.dpath, self.imgslist[index])
        color_image = cv2.imread(ipath)
        depth_image = cv2.imread(dpath)
        d, d, d = cv2.split(depth_image)
        if self.mode == 1:
            rgbtrans = transforms.Compose(
                ([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)]))
            color_image = Image.fromarray(color_image)
            color_image = rgbtrans(color_image)
            color_image = np.asarray(color_image)
        b, g, r = cv2.split(color_image)
        img = cv2.merge([r, g, b, d])
        img = cv2.resize(img, (224, 224))
        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        # if(len(label) != 1248):
        #     print(filename)
        return img, label, filename


class MultiLabelRGBDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        img = Image.open(ipath)
        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".txt")
        label = np.loadtxt(annotation, dtype=np.int64)
        return img, label, filename


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, loss_fn=criterion)
        train_loss, train_acc = train_evaluate(model)
        test_loss, test_acc = test(model)
