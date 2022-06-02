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
import torch.nn as nn
import time
import copy
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import os
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
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


class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, dpath, imgslist, annotationpath, transforms=None, mode=0):
        self.imgslist = imgslist
        # print(np.size(os.listdir(imgspath)))
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
        return img, label[:384], label[384:768], label[768:1152], label[1152:1248]


class MultiLabelRGBataSet(torch.utils.data.Dataset):
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


trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    transforms.ToTensor()  # divides by 255
]))

# alist = os.listdir('/home/apline/Desktop/Data/RGB/Images/Labeled/')

rgbdir = r'F:\34294\RGB'
depthdir = r'F:\34294\Depth'
labeldir = r'F:\34294\yoliclabel'

# rgbdir = '/media/apline/2TSSD/34294/RGB'
# depthdir = '/media/apline/2TSSD/34294/Depth'
# labeldir = '/media/apline/2TSSD/34294/yoliclabel'

alist = os.listdir(rgbdir)
x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)

# train = MultiLabelRGBDataSet(r'D:\23308\RGB',
#                           x_train,r'D:\23308\yoliclabel',trans)

# test = MultiLabelRGBDataSet(r'D:\23308\RGB',
#                           x_test,r'D:\23308\yoliclabel',trans)

train = MultiLabelDataSet(rgbdir, depthdir,
                          x_train, labeldir, trans, 1)

test = MultiLabelDataSet(rgbdir, depthdir,
                         x_test, labeldir, trans, 0)

# x_train ,x_test = train_test_split(rgbimage.imgs,test_size=0.3,random_state=1)

# train = RealsenseDataSet(x_train,val_trans)

# test = RealsenseDataSet(x_test,val_trans)
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=10)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=10)

# (data, target) =iter(train_loader).next()
# exit()
# from resnet import resnet20, resnet56

# model = models.resnet18()
# model = resnet56()
# print(model)
# best_model_wts = copy.deepcopy(model.state_dict())
# torch.save(best_model_wts, os.path.join(os.getcwd(), "resnet20_fp16_34294rgbd_1248output.pth.tar"))
# model.conv1 = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
# model.linear = nn.Linear(64, 1248)

# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(512,1248)
from branch_network import branch_network
model = branch_network(branch1_classes=384, branch2_classes=384, branch3_classes=384, branch4_classes=96)

from BranchMobileNetV2 import BranchMobileNetV2
model = BranchMobileNetV2()
# model = models.mobilenet_v2(pretrained=True)
# model.classifier[1] = nn.Linear(1280, 1248)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
save_name = 'multibranch_molnet_34294_fp16'


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

# teacher = VGG16()

# model = torch.nn.DataParallel(model)

# checkpoint = torch.load(os.path.join(os.getcwd(), "mobilenetv2_fp16_34294rgbd_1248_normal_batch32.pth.tar"))
# model.load_state_dict(checkpoint)

# model.module.fc=nn.Linear(2048,968)
# print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
scaler = GradScaler()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler = StepLR(optimizer,step_size = 50,gamma=0.1)
# w = [0.23, 1.38, 1.22, 1.66, 0.95, 0.90, 0.52, 0.28, 0.30, 4.66, 19, 0.005]
w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pWeights = list(map(lambda x: x * 4, w * 32)) + list(map(lambda x: x * 1, w * 32)) + list(
    map(lambda x: x * 1, w * 32)) + list(map(lambda x: x * 1, w * 8))
pWeights = torch.FloatTensor(pWeights).cuda()
# criterion = nn.MultiLabelSoftMarginLoss(pWeights)
criterion = nn.MultiLabelSoftMarginLoss()



BCEloss = torch.nn.BCEWithLogitsLoss(pWeights)
# bceloss = torch.nn.BCEWithLogitsLoss()

def get_weighted_loss(weights):

    def weighted_loss(y_pred, y_true):
        # output = torch.sigmoid(output)
        # print(np.shape((weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * BCEloss(y_pred, y_true.float())))
        return torch.mean(torch.sum(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * BCEloss(y_pred, y_true.float()), dim=-1))

    return weighted_loss


# weight = np.load('newweight.npy')
# weight = torch.tensor(weight).cuda()
# criterion = get_weighted_loss(weight)
# criterion = nn.MultiLabelSoftMarginLoss()


def pred_acc(original, predicted, errornum):
    pred = torch.round(predicted).detach().numpy().astype(np.int64)
    orig = original.detach().numpy()
    num = 0
    enum = 0
    normal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    for cell in range(0, 1248, 12):
        # print(cell, cell + 12)
        if (orig[cell:cell + 12] == pred[cell:cell + 12]).all():

            # print(pred[cell:cell + 12])
            num = num + 1
        else:
            if not (orig[cell:cell + 12] == normal).all() and not (pred[cell:cell + 12] == normal).all():
                if errornum == -1:
                    enum = enum + 1
                # else:
                #     for each in range(0, 12):
                #         if (orig[cell:cell + 12][each] == pred[cell:cell + 12][each]) and orig[cell:cell + 12][each] != 0:
                #             # num = num + 1
                #             pass
    if errornum == -1:
        # print(num + enum)
        return (num + enum) / 104
    if errornum == 0:
        # print(num)
        return num / 104

    # if errornum==0:
    # return torch.round(predicted).eq(original).sum().numpy()/len(original)


def train(epoch, model, loss_fn):
    model.train()
    for batch_idx, (data, target1, target2, target3, target4) in enumerate(train_loader):
        if args.cuda:
            data, target1, target2, target3, target4 = data.cuda(), target1.cuda(), target2.cuda(), target3.cuda(), target4.cuda()
        optimizer.zero_grad()
        with autocast():
            branch1_out, branch2_out, branch3_out, branch4_out = model(data)

            loss1_out = loss_fn(branch1_out, target1)
            loss2_out = loss_fn(branch2_out, target2)
            loss3_out = loss_fn(branch3_out, target3)
            loss4_out = loss_fn(branch4_out, target4)
            loss = loss1_out + loss2_out + loss3_out + loss4_out
            # print(loss)

        # loss = F.softmax(output, target)
        # loss = F.cross_entropy(output, target)
        # loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        # pred = output.data.max(1, keepdim=True)[1]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


best_correct = -999


def train_evaluate(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_acc5 = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target1, target2, target3, target4) in enumerate(train_loader):
            if args.cuda:
                data, target1, target2, target3, target4 = data.cuda(), target1.cuda(), target2.cuda(), target3.cuda(), target4.cuda()

            branch1_out, branch2_out, branch3_out, branch4_out = model(data)
            loss1_out = criterion(branch1_out, target1)
            loss2_out = criterion(branch2_out, target2)
            loss3_out = criterion(branch3_out, target3)
            loss4_out = criterion(branch4_out, target4)
            loss = loss1_out + loss2_out + loss3_out + loss4_out
            all_out = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), 1)
            output = torch.sigmoid(all_out)
            target = torch.cat((target1, target2, target3, target4), 1)
            # print(output)
            acc_ = []
            for each_image, d in enumerate(output):
                acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), 0)
                acc_.append(acc)

            acc_5 = []
            for each_image, d in enumerate(output):
                acc = pred_acc(torch.Tensor.cpu(target[each_image]), torch.Tensor.cpu(d), -1)
                acc_5.append(acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_).mean())
            running_acc5.append(np.asarray(acc_5).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_acc5 = np.asarray(running_acc5).mean()
    print('\nTrain set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), 2ACC: ({:.4f}%)\n'.format(
        total_batch_loss, len(train_loader.dataset), total_batch_acc, total_batch_acc5))
    now_correct = total_batch_acc
    if best_correct < now_correct:
        best_correct = now_correct
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,
                   os.path.join(os.getcwd(), save_name + ".pth.tar"))
        print("New weight!")
    return total_batch_loss, total_batch_acc


def test(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_acc5 = []
    global best_correct
    with torch.no_grad():
        for batch_idx, (data, target1, target2, target3, target4) in enumerate(test_loader):
            if args.cuda:
                data, target1, target2, target3, target4 = data.cuda(), target1.cuda(), target2.cuda(), target3.cuda(), target4.cuda()

            branch1_out, branch2_out, branch3_out, branch4_out = model(data)
            loss1_out = criterion(branch1_out, target1)
            loss2_out = criterion(branch2_out, target2)
            loss3_out = criterion(branch3_out, target3)
            loss4_out = criterion(branch4_out, target4)
            loss = loss1_out + loss2_out + loss3_out + loss4_out
            all_out = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), 1)
            output = torch.sigmoid(all_out)
            target = torch.cat((target1, target2, target3, target4), 1)

            acc_ = []
            for i, d in enumerate(output):
                acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0)
                acc_.append(acc)

            acc_5 = []
            for i, d in enumerate(output):
                acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), -1)
                acc_5.append(acc)
            running_loss.append(loss.item())
            running_acc.append(np.asarray(acc_).mean())
            running_acc5.append(np.asarray(acc_5).mean())
    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()
    total_batch_acc5 = np.asarray(running_acc5).mean()
    print('\nTest set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), 2classAcc: ({:.4f}%)\n'.format(
        total_batch_loss, len(test_loader.dataset), total_batch_acc, total_batch_acc5))

    return total_batch_loss, total_batch_acc


if __name__ == '__main__':
    # test_loss, test_acc = test(model)
    print(save_name)
    all_train_loss = []
    all_train_acc = []
    all_test_loss = []
    all_test_acc = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, loss_fn=criterion)
        train_loss, train_acc = train_evaluate(model)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)
        test_loss, test_acc = test(model)
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)

    list_res = []
    for i in range(len(all_train_loss)):
        list_res.append([all_train_loss[i], all_train_acc[i], all_test_loss[i], all_test_acc[i]])

    column_name = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    csv_name = save_name + '.csv'
    xml_df = pd.DataFrame(list_res, columns=column_name)
    xml_df.to_csv(csv_name, index=None)
