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


trans = transforms.Compose(([
    # transforms.Resize((224,224)),
    transforms.ToTensor()  # divides by 255
]))

# alist = os.listdir('/home/apline/Desktop/Data/RGB/Images/Labeled/')

rgbdir = r'C:\Users\Kai\Desktop\AlpineData\RGB'
depthdir = r'C:\Users\Kai\Desktop\AlpineData\Depth'
labeldir = r'C:\Users\Kai\Desktop\AlpineData\yoliclabel'
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
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False)

# (data, target) =iter(train_loader).next()
# exit()
#
# from FPNnet import MobileNetV2_dynamicFPN
#
# model = MobileNetV2_dynamicFPN()
# # model.fc = nn.Linear(2048, 1248)
# # model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # # model = models.alexnet()
#
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(1280,1248)
model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#
#
# # model = models.shufflenet_v2_x2_0()
# # model.fc=nn.Linear(2048,1248)
# # model.conv1[0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#
# # from mixnet import MixNet
# # model = MixNet(net_type='mixnet_s', input_size=224, num_classes=1248)
# # model.stem_conv[0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# # model.classifier[1] = nn.Linear(1280,1248)
# # model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# # model.fc=nn.Linear(4096,1248)
# # model = models.vgg16()
# # model.classifier[6] = nn.Linear(4096,22)
print(model)
# teacher = VGG16()

model = torch.nn.DataParallel(model)

# checkpoint = torch.load(os.path.join(os.getcwd(),"resnet50_23308.pth.tar"))
# model.load_state_dict(checkpoint)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())

# optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler = StepLR(optimizer,step_size = 50,gamma=0.1)

criterion = nn.MultiLabelSoftMarginLoss()


def pred_acc(original, predicted, errornum):
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
                if errornum == -1:
                    enum = enum + 1
                else:
                    for j in range(0, 12):
                        if (orig[cell:cell + 12][j] == pred[cell:cell + 12][j]) and not orig[cell:cell + 12][j] == 0:
                            # num=num+1
                            pass
    if errornum == -1:
        return (num + enum) / 104
    if errornum == 0:
        return num / 104
    if errornum == 1:
        if (num + 1) >= 104:
            return 1
        return (num + 1) / 104
    if errornum == 2:
        if (num + 2) >= 104:
            return 1
        return (num + 2) / 104
    if errornum == 3:
        if (num + 3) >= 104:
            return 1
        return (num + 3) / 104
    if errornum == 4:
        if (num + 4) >= 104:
            return 1
        return (num + 4) / 104
    if errornum == 5:
        if (num + 5) >= 104:
            return 1
        return (num + 5) / 104

    # if errornum==0:
    #     return torch.round(predicted).eq(original).sum().numpy()/len(original)



def train(epoch, model, loss_fn):
    model.train()
    correct = 0
    # i =0
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        # loss = F.softmax(output, target)
        # loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # pred = output.data.max(1, keepdim=True)[1]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


trainLoss_list = []


def train_evaluate(model):
    model.eval()
    plt_score_val = []
    running_loss = []
    running_acc = []
    running_acc1 = []
    running_acc2 = []
    running_acc3 = []
    running_acc4 = []
    running_acc5 = []
    global best_correct
    for batch_idx, (data, target, filenames) in enumerate(train_loader):
        # data = data[1].numpy()
        # plt.imshow(data)
        # exit
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        output = torch.sigmoid(output)
        # print(output)
        acc_ = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0)
            acc_.append(acc)
        acc_1 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 1)
            acc_1.append(acc)
        acc_2 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 2)
            acc_2.append(acc)
        acc_3 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 3)
            acc_3.append(acc)
        acc_4 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 4)
            acc_4.append(acc)
        # acc_5 = []
        # for i,d in enumerate(output):
        #     acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d),-1)
        #     acc_5.append(acc)
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        running_acc1.append(np.asarray(acc_1).mean())
        running_acc2.append(np.asarray(acc_2).mean())
        running_acc3.append(np.asarray(acc_3).mean())
        running_acc4.append(np.asarray(acc_4).mean())
        # running_acc5.append(np.asarray(acc_5).mean())
        total_batch_loss = np.asarray(running_loss).mean()
        trainLoss_list.append(total_batch_loss)
        total_batch_acc = np.asarray(running_acc).mean()
        total_batch_acc1 = np.asarray(running_acc1).mean()
        total_batch_acc2 = np.asarray(running_acc2).mean()
        total_batch_acc3 = np.asarray(running_acc3).mean()
        total_batch_acc4 = np.asarray(running_acc4).mean()
        # total_batch_acc5 = np.asarray(running_acc5).mean()
        total_batch_acc5 = 0
    print(
        '\nTrain set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Acc_1: ({:.4f}%), Acc_2: ({:.4f}%), Acc_3: ({:.4f}%), Acc_4: ({:.4f}%), 2ACC: ({:.4f}%)\n'.format(
            total_batch_loss, len(train_loader.dataset), total_batch_acc, total_batch_acc1, total_batch_acc2,
            total_batch_acc3, total_batch_acc4, total_batch_acc5))

    return total_batch_loss, total_batch_acc


best_correct = -999

testLoss_list = []


def test(model):
    model.eval()
    running_loss = []
    running_acc = []
    running_acc1 = []
    running_acc2 = []
    running_acc3 = []
    running_acc4 = []
    running_acc5 = []
    global best_correct
    for batch_idx, (data, target, filenames) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        output = torch.sigmoid(output)
        acc_ = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 0)
            acc_.append(acc)
        acc_1 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 1)
            acc_1.append(acc)
        acc_2 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 2)
            acc_2.append(acc)
        acc_3 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 3)
            acc_3.append(acc)
        acc_4 = []
        for i, d in enumerate(output):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d), 4)
            acc_4.append(acc)
        # acc_5 = []
        # for i,d in enumerate(output):
        #     acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d),-1)
        #     acc_5.append(acc)
        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        running_acc1.append(np.asarray(acc_1).mean())
        running_acc2.append(np.asarray(acc_2).mean())
        running_acc3.append(np.asarray(acc_3).mean())
        running_acc4.append(np.asarray(acc_4).mean())
        # running_acc5.append(np.asarray(acc_5).mean())
        total_batch_loss = np.asarray(running_loss).mean()
        testLoss_list.append(total_batch_loss)
        total_batch_acc = np.asarray(running_acc).mean()
        total_batch_acc1 = np.asarray(running_acc1).mean()
        total_batch_acc2 = np.asarray(running_acc2).mean()
        total_batch_acc3 = np.asarray(running_acc3).mean()
        total_batch_acc4 = np.asarray(running_acc4).mean()
        total_batch_acc5 = 0
        # total_batch_acc5 = np.asarray(running_acc5).mean()
    print(
        '\nTest set: total_batch_loss: {:.4f}, total imgs: {} , Acc: ({:.4f}%), Acc_1: ({:.4f}%), Acc_2: ({:.4f}%), Acc_3: ({:.4f}%), Acc_4: ({:.4f}%), 2classAcc: ({:.4f}%)\n'.format(
            total_batch_loss, len(test_loader.dataset), total_batch_acc, total_batch_acc1, total_batch_acc2,
            total_batch_acc3, total_batch_acc4, total_batch_acc5))
    now_correct = total_batch_acc1
    if best_correct < now_correct:
        print("New weight!")
        best_correct = now_correct
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(os.getcwd(), "resnet50_28406rgbd_1248output.pth.tar"))
    return total_batch_loss, total_batch_acc


scores = np.array([])
val_scores = []
for epoch in range(1, args.epochs + 1):
    # train(epoch, model, loss_fn=criterion)
    # train_evaluate(model)
    test(model)
    # optimizer.step()

print("The mean accuracy: {}".format(np.mean(scores)))
print("The STD : {}: ".format(np.std(scores)))
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()