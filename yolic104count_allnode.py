# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:43:02 2021

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
import os
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from FPNnet import MobileNetV2_dynamicFPN

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, imgspath, dpath, imgslist, annotationpath, transforms=None):
        self.imgslist = imgslist
        self.imgspath = imgspath
        self.dpath = dpath
        self.transform = transforms
        self.annotationpath = annotationpath
        # print(annotationpath)

    def __len__(self):
        return len(self.imgslist)

    def __getitem__(self, index):
        ipath = os.path.join(self.imgspath, self.imgslist[index])
        dpath = os.path.join(self.dpath, self.imgslist[index])
        color_image = cv2.imread(ipath)
        depth_image = cv2.imread(dpath)
        d, d, d = cv2.split(depth_image)
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
        label = np.reshape(label, (13, 8, 12))
        label = np.transpose(label, (2, 0, 1))
        # print(labels)
        return img, label, filename


trans = transforms.Compose(([

    transforms.ToTensor()  # divides by 255
]))

# alist = os.listdir('/home/apline/Desktop/Data/RGB/Images/Labeled/')
rgb_dir = '/home/kai/Desktop/RGB'
depth_dir = '/home/kai/Desktop/Depth'
label_dir = '/home/kai/Desktop/yoliclabel'

img_list = os.listdir(rgb_dir)
x_train, x_test = train_test_split(img_list, test_size=0.3, random_state=2)

test = MultiLabelDataSet(rgb_dir, depth_dir,
                         x_test, label_dir, trans)

train = MultiLabelDataSet(rgb_dir, depth_dir,
                          x_train, label_dir, trans)


test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False)


from mobilenet_ import mobilenet_v2
model = mobilenet_v2()

# model = torch.nn.DataParallel(model)
checkpoint = torch.load("/home/kai/Desktop/AlpineProject/mobilenet_v2con1.pth.tar")
title_name = 'mobilenet_v2con'
print(title_name)
model.load_state_dict(checkpoint)

if args.cuda:
    model.cuda()


def pred_cm(original, predicted):
    global cm_true
    global cm_pred
    global cm2_true
    global cm2_pred
    orig = original.detach().numpy()

    pred_sigmoid = predicted.detach().numpy()

    pred_sigmoid = np.reshape(pred_sigmoid, (12 * 13 * 8, 1)).flatten()
    orig = np.reshape(orig, (12 * 13 * 8, 1)).flatten()
    # pred = torch.round(predicted).detach().numpy()
    pred = np.where(pred_sigmoid > 0, 1, 0)

    normal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    for i in range(0, 1248, 12):
    # for i in range(0, 384, 12):  # 2-6m
    # for i in range(384, 768, 12): #  1-2m
    # for i in range(768, 1152, 12):  #  0-1m
        cell_ture = np.argmax(orig[i:i + 12])

        cell_pred = np.argmax(pred_sigmoid[i:i + 12])
        # cell_pred = np.argmax(pred[i:i + 12])

        if cell_pred == 11 and not (pred[i:i + 12] == normal).all():
            # print("find", pred[i:i + 12])
            cell_pred = np.argsort(pred_sigmoid[i:i + 12])[-2]
        #     # print(cell_pred)
        # if cell_pred == 11:
        #     print(pred[i:i + 12])
        # if cell_pred ==1:
        #     print(pred[i:i+12])

        for j in range(0, 11):
            if orig[i:i + 12][j] == pred[i:i + 12][j] and orig[i:i + 12][j] != 0:
                # print(orig[i:i + 12])
                # print(pred[i:i + 12])
                cell_ture = j
                cell_pred = j
                break
        # if cell_pred == 11: cell_pred = 10
        # if cell_ture == 11: cell_ture = 10

        cm_true.append(cell_ture)
        cm_pred.append(cell_pred)

        if (orig[i:i + 12] == normal).all():
            cm2_true.append(1)
        else:
            cm2_true.append(0)

        if (pred[i:i + 12] == normal).all():

            cm2_pred.append(1)
        else:
            cm2_pred.append(0)
            # if(cell_ture==10):
        #     print(cell_ture)
        #     print(cell_pred)
        #     print(" ")


def pred_cost1(original, predicted):
    pred = torch.where(predicted > 0.5, 1, 0).detach().numpy()
    original = original.detach().numpy()

    FN = 0
    FP = 0
    for i in range(0, 1152, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = 2 * FN + FP
    return cost


def pred_cost2(original, predicted):
    pred = torch.where(predicted > 0.5, 1, 0).detach().numpy()
    original = original.detach().numpy()
    pred = np.reshape(pred, (12 * 13 * 8, 1)).flatten()
    original = np.reshape(original, (12 * 13 * 8, 1)).flatten()
    FN = 0
    FP = 0
    cost = 0
    for i in range(0, 384, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = 2 * FN + 1 * FP
    FN = 0
    FP = 0

    for i in range(384, 768, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 4 * FN + 2 * FP
    FN = 0
    FP = 0

    for i in range(768, 1152, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 8 * FN + 4 * FP
    return cost


def pred_cost3(original, predicted):
    pred = torch.where(predicted > 0.5, 1, 0).detach().numpy()
    original = original.detach().numpy()

    FN = 0
    FP = 0
    cost = 0
    for i in range(0, 96, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = 2 * FN + FP
    FN = 0
    FP = 0

    for i in range(96, 240, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 2.5 * FN + 1.5 * FP
    FN = 0
    FP = 0

    for i in range(240, 384, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 3 * FN + 2 * FP
    FN = 0
    FP = 0

    for i in range(384, 576, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 4 * FN + 2.5 * FP
    FN = 0
    FP = 0

    for i in range(576, 768, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 4.5 * FN + 3 * FP
    FN = 0
    FP = 0

    for i in range(768, 960, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 5 * FN + 3.5 * FP
    FN = 0
    FP = 0

    for i in range(960, 1152, 12):
        if pred[i:i + 12][11] != original[i:i + 12][11]:
            if pred[i:i + 12][11] == 1:
                FN = FN + 1
            else:
                FP = FP + 1
    cost = cost + 5.5  * FN + 4 * FP

    return cost


best_correct = -999

testLoss_list = []
cm_true = []
cm_pred = []
cm2_true = []
cm2_pred = []
cost1_all = []
cost2_all = []
cost3_all = []
filename_list = []
count = 0


def test(model):
    model.eval()
    global best_correct
    global count
    normal = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 104)
    with torch.no_grad():
        for data, target, filename in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            output = torch.permute(output, (0, 2, 3, 1))
            target = torch.permute(target, (0, 2, 3, 1))
            # avg = torch.zeros_like(outputs[0]).cuda()
            # for output in outputs:
            #     output = torch.sigmoid(output)
            #     avg = avg + output
            # avg = avg / 5
            # output = avg.unsqueeze(0)

            cost2 = pred_cost2(torch.Tensor.cpu(target[0]), torch.Tensor.cpu(output[0]))

            cost2_all.append(cost2)

            pred_cm(torch.Tensor.cpu(target[0]), torch.Tensor.cpu(output[0]))

        # print("mean:",np.asarray(cost1_all).mean())
        # print("std:",np.asarray(cost1_all).std())
        # print("max:",np.asarray(cost1_all).max())

        print("mean:", np.asarray(cost2_all).mean())
        print("std:", np.asarray(cost2_all).std())
        print("max:", np.asarray(cost2_all).max())

        # print("mean:",np.asarray(cost3_all).mean())
        # print("std:",np.asarray(cost3_all).std())
        # print("max:",np.asarray(cost3_all).max())


import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm_normalize = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_normalize, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm_normalize.max() / 2.
    for i, j in itertools.product(range(cm_normalize.shape[0]), range(cm_normalize.shape[1])):
        plt.text(j, i-0.1, format(cm_normalize[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
        plt.text(j, i+0.2, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_normalize[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


# from shutil import copyfile
test(model)
# 
class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
               "TrafficSign", 'Normal']
# class_names = ["Bump", "Column", "Dent", "Fence", "Creature", "Vehicle", "Wall", "Weed", "ZebraCrossing", "TrafficCone",
#                "Normal"]

print(metrics.classification_report(cm_true, cm_pred, target_names=class_names, digits=4))

class2_names = ["Risk", "Normal"]
print(metrics.classification_report(cm2_true, cm2_pred, target_names=class2_names, digits=4))

matrix = confusion_matrix(cm_true, cm_pred)
matrix2 = confusion_matrix(cm2_true, cm2_pred)
plt.figure(figsize=(10, 10))

plot_confusion_matrix(matrix, classes=class_names, normalize=True, title=title_name)
plt.show()
# plt.savefig( 'Confusion Matrix.png')
plt.close()

plt.figure(figsize=(5, 5))

plot_confusion_matrix(matrix2, classes=class2_names, normalize=True, title=title_name)
plt.show()
# plt.savefig( 'oConfusion Matrix .png')
plt.close()

# from shutil import copyfile
# target = os.path.join(os.getcwd(),"check")
# os.makedirs(target)
# for file in filename_list:
#     path = os.path.join(rgbdir, file+".png")
#     new =  os.path.join(target, file+".png")
#     copyfile(path, new)