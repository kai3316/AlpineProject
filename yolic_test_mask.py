#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:39:10 2020

@author: Su
"""
from multiprocessing import freeze_support
from sklearn.metrics import confusion_matrix
from sklearn import metrics
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
import pandas as pd
import os

import moblienet_mask
import yolicNet
from mobilenext import MobileNeXt
from ownTiny0615 import mobilenet_v2
from net0613 import mbv2_ca0613

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
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


# class MultiLabelDataSet(torch.utils.data.Dataset):
#     def __init__(self, imgspath, dpath, imgslist, annotationpath, transforms=None, mode=0):
#         self.imgslist = imgslist
#         self.imgspath = imgspath
#         self.dpath = dpath
#         self.transform = transforms
#         self.annotationpath = annotationpath
#         self.mode = mode
#         # print(annotationpath)
#
#     def __len__(self):
#         return len(self.imgslist)
#
#     def __getitem__(self, index):
#         ipath = os.path.join(self.imgspath, self.imgslist[index])
#         dpath = os.path.join(self.dpath, self.imgslist[index])
#         color_image = cv2.imread(ipath)
#         depth_image = cv2.imread(dpath)
#         d, d, d = cv2.split(depth_image)
#         if self.mode == 1:
#             rgbtrans = transforms.Compose(
#                 ([transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)]))
#             color_image = Image.fromarray(color_image)
#             color_image = rgbtrans(color_image)
#             color_image = np.asarray(color_image)
#         b, g, r = cv2.split(color_image)
#         img = cv2.merge([r, g, b, d])
#         img = cv2.resize(img, (224, 224))
#         # print(ipath)
#         if self.transform is not None:
#             img = self.transform(img)
#         (filename, extension) = os.path.splitext(ipath)
#         filename = os.path.basename(filename)
#         annotation = os.path.join(self.annotationpath, filename + ".txt")
#         label = np.loadtxt(annotation, dtype=np.int64)
#         # if(len(label) != 1248):
#         #     print(filename)
#         return img, label, filename


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
        # flip the image horizontally with probability 0.5
        (filename, extension) = os.path.splitext(ipath)
        filename = os.path.basename(filename)
        annotation = os.path.join(self.annotationpath, filename + ".npy")
        labelimg = np.load(annotation)
        # numpy to tensor
        # labelimg = torch.from_numpy(labelimg)
        # resize the labelimg(12,480,848) to (12,56, 56)

        # print(labelimg.shape)
        # if np.random.random() > 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     labelimg = np.fliplr(labelimg).copy()
            # print(labelimg.shape)

        # print(ipath)
        if self.transform is not None:
            img = self.transform(img)
        # print(labelimg.shape, img.shape)
        return img, labelimg, filename


trans = transforms.Compose(([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # divides by 255
    transforms.Normalize([0.4328, 0.4387, 0.4203], [0.2046, 0.2025, 0.2172])
]))

# alist = os.listdir('/home/apline/Desktop/Data/RGB/Images/Labeled/')

# rgbdir = r'C:\Users\3090\Desktop\34294\RGB'
# depthdir = r'C:\Users\3090\Desktop\34294\Depth'
# labeldir = r'C:\Users\3090\Desktop\34294\yoliclabel'

rgb_dir = '/home/kai/Desktop/data_noflip/RGB_noflip'
# depth_dir = '/home/kai/Desktop/Depth'
label_dir = '/home/kai/Desktop/data_noflip/yolicmask'

alist = os.listdir(rgb_dir)
x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)

train = MultiLabelRGBataSet(rgb_dir, x_train, label_dir, trans)

test = MultiLabelRGBataSet(rgb_dir, x_test, label_dir, trans)

# train = MultiLabelDataSet(rgb_dir, depth_dir,
#                           x_train, label_dir, trans, 1)
#
# test = MultiLabelDataSet(rgb_dir, depth_dir,
#                          x_test, label_dir, trans, 0)

# x_train ,x_test = train_test_split(rgbimage.imgs,test_size=0.3,random_state=1)

# train = RealsenseDataSet(x_train,val_trans)

# test = RealsenseDataSet(x_test,val_trans)
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=8)

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
# import timm
# model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
# model = models.mobilenet_v3_large(pretrained=True)
# model.classifier[3] = nn.Linear(1280, 1248, bias=True)
# print(model)
# model.fc = nn.Linear(in_features=2048, out_features=1248, bias=True)
# model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.classifier[1] = nn.Linear(1280, 1248)
# checkpoint = torch.load(os.path.join(os.getcwd(),"mobilenetv2_fp16_34294rgbd_1248_position_sign_weight.pth.tar"))
# model.load_state_dict(checkpoint)
# model.classifier[1] = nn.Linear(1280, 96)
# print(model)
# model.head = nn.Linear(in_features=1024, out_features=1248, bias=True)
# a = model.patch_embed.proj = nn.Conv2d(4, 128, kernel_size=(4, 4), stride=(4, 4))
# from convnext import *
# model = convnext_large()
# print(model)
import torchvision.models as models

# model = mbv2_ca0613()  # resnet.resnet18()#
features_map = (28, 28)
model = yolicNet.mobilenet_v2()
# load the pretrained weights
model.load_state_dict(torch.load("/home/kai/Desktop/AlpineProject/mobile_own7777_0613.pth.tar"))
# model = models.mobilenet_v2()
# model = MobileNeXt(num_classes=1248, width_mult=1.0, identity_tensor_multiplier=1.0)
# model = mobilenet_v2()
# model.classifier[1] = nn.Linear(1280, 1248)
# model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
save_name = 'tinyown0615-[2,16,2,2,3][4,32,2,1,3][4,64,1,1,3][4,128,1,1,3]'
# print(model)
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


# model.module.fc=nn.Linear(2048,968)
# print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters())
# scaler = GradScaler()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler = StepLR(optimizer,step_size = 50,gamma=0.1)
# w = [0.23, 1.38, 1.22, 1.66, 0.95, 0.90, 0.52, 0.28, 0.30, 4.66, 19, 0.005]
# w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Weights = list(map(lambda x: x * 1, w * 32)) + list(map(lambda x: x * 2, w * 32)) + list(
#     map(lambda x: x * 4, w * 32)) + list(map(lambda x: x * 8, w * 8))
# Weights = torch.FloatTensor(Weights).cuda()
# criterion = nn.MultiLabelSoftMarginLoss(Weights)
criterion = nn.MultiLabelSoftMarginLoss()

# bceloss = torch.nn.BCEWithLogitsLoss(Weights)
#
#
# def get_weighted_loss(weights):
#     def weighted_loss(y_pred, y_true):
#         return torch.mean(
#             (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** y_true) * bceloss(y_pred, y_true.float()))
#
#     return weighted_loss
#
#
# weight = np.load('newweight.npy')
# weight = torch.tensor(weight).cuda()
# criterion = get_weighted_loss(weight)

def pred_cm(original, predicted):
    global cm_true
    global cm_pred
    global cm2_true
    global cm2_pred
    orig = original.detach().numpy()

    pred_sigmoid = predicted.detach().numpy()



    # pred = torch.round(predicted).detach().numpy()
    pred = np.where(pred_sigmoid > 0.5, 1, 0)
    orig, pred, pred_sigmoid = masktoLabel(orig, pred, pred_sigmoid)
    orig = np.concatenate(orig).tolist()
    pred = np.concatenate(pred).tolist()
    pred_sigmoid = np.concatenate(pred_sigmoid).tolist()
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

points_list = [(288, 166), (322, 166), (356, 166), (390, 166), (424, 166), (458, 166), (492, 166), (526, 166),
               (220, 200), (254, 200), (288, 200), (322, 200), (356, 200), (390, 200), (424, 200), (458, 200),
               (492, 200), (526, 200), (560, 200), (594, 200),
               (220, 234), (254, 234), (288, 234), (322, 234), (356, 234), (390, 234), (424, 234), (458, 234),
               (492, 234), (526, 234), (560, 234), (594, 234), (628, 234),
               (254, 268), (288, 268), (322, 268), (356, 268), (390, 268), (424, 268), (458, 268), (492, 268),
               (526, 268), (560, 268), (594, 268), (628, 268),
               (0, 268), (53, 268), (106, 268), (159, 268), (212, 268), (265, 268), (318, 268), (371, 268), (424, 268),
               (477, 268), (530, 268), (583, 268), (636, 268), (689, 268), (742, 268), (795, 268),
               (0, 321), (53, 321), (106, 321), (159, 321), (212, 321), (265, 321), (318, 321), (371, 321), (424, 321),
               (477, 321), (530, 321), (583, 321), (636, 321), (689, 321), (742, 321), (795, 321), (848, 321),
               (0, 374), (53, 374), (106, 374), (159, 374), (212, 374), (265, 374), (318, 374), (371, 374), (424, 374),
               (477, 374), (530, 374), (583, 374), (636, 374), (689, 374), (742, 374), (795, 374), (848, 374),
               (0, 427), (53, 427), (106, 427), (159, 427), (212, 427), (265, 427), (318, 427), (371, 427), (424, 427),
               (477, 427), (530, 427), (583, 427), (636, 427), (689, 427), (742, 427), (795, 427), (848, 427),
               (0, 480), (53, 480), (106, 480), (159, 480), (212, 480), (265, 480), (318, 480), (371, 480), (424, 480),
               (477, 480), (530, 480), (583, 480), (636, 480), (689, 480), (742, 480), (795, 480), (848, 480),
               (184, 0), (244, 0), (304, 0), (364, 0), (424, 0), (484, 0), (544, 0), (604, 0), (244, 60), (304, 60),
               (364, 60), (424, 60), (484, 60), (544, 60), (604, 60), (664, 60)]

box_list = [[points_list[0], points_list[11]], [points_list[1], points_list[12]], [points_list[2], points_list[13]],
            [points_list[3], points_list[14]], [points_list[4], points_list[15]],
            [points_list[5], points_list[16]], [points_list[6], points_list[17]], [points_list[7], points_list[18]],
            [points_list[8], points_list[21]], [points_list[9], points_list[22]],
            [points_list[10], points_list[23]], [points_list[11], points_list[24]], [points_list[12], points_list[25]],
            [points_list[13], points_list[26]], [points_list[14], points_list[27]],
            [points_list[15], points_list[28]], [points_list[16], points_list[29]], [points_list[17], points_list[30]],
            [points_list[18], points_list[31]], [points_list[19], points_list[32]],
            [points_list[20], points_list[33]], [points_list[21], points_list[34]], [points_list[22], points_list[35]],
            [points_list[23], points_list[36]], [points_list[24], points_list[37]],
            [points_list[25], points_list[38]], [points_list[26], points_list[39]], [points_list[27], points_list[40]],
            [points_list[28], points_list[41]], [points_list[29], points_list[42]],
            [points_list[30], points_list[43]], [points_list[31], points_list[44]], [points_list[45], points_list[62]],
            [points_list[46], points_list[63]], [points_list[47], points_list[64]],
            [points_list[48], points_list[65]], [points_list[49], points_list[66]], [points_list[50], points_list[67]],
            [points_list[51], points_list[68]], [points_list[52], points_list[69]],
            [points_list[53], points_list[70]], [points_list[54], points_list[71]], [points_list[55], points_list[72]],
            [points_list[56], points_list[73]], [points_list[57], points_list[74]],
            [points_list[58], points_list[75]], [points_list[59], points_list[76]], [points_list[60], points_list[77]],
            [points_list[61], points_list[79]], [points_list[62], points_list[80]],
            [points_list[63], points_list[81]], [points_list[64], points_list[82]], [points_list[65], points_list[83]],
            [points_list[66], points_list[84]], [points_list[67], points_list[85]],
            [points_list[68], points_list[86]], [points_list[69], points_list[87]], [points_list[70], points_list[88]],
            [points_list[71], points_list[89]], [points_list[72], points_list[90]],
            [points_list[73], points_list[91]], [points_list[74], points_list[92]], [points_list[75], points_list[93]],
            [points_list[76], points_list[94]], [points_list[78], points_list[96]],
            [points_list[79], points_list[97]], [points_list[80], points_list[98]], [points_list[81], points_list[99]],
            [points_list[82], points_list[100]], [points_list[83], points_list[101]],
            [points_list[84], points_list[102]], [points_list[85], points_list[103]],
            [points_list[86], points_list[104]], [points_list[87], points_list[105]],
            [points_list[88], points_list[106]],
            [points_list[89], points_list[107]], [points_list[90], points_list[108]],
            [points_list[91], points_list[109]], [points_list[92], points_list[110]],
            [points_list[93], points_list[111]],
            [points_list[95], points_list[113]], [points_list[96], points_list[114]],
            [points_list[97], points_list[115]], [points_list[98], points_list[116]],
            [points_list[99], points_list[117]],
            [points_list[100], points_list[118]], [points_list[101], points_list[119]],
            [points_list[102], points_list[120]], [points_list[103], points_list[121]],
            [points_list[104], points_list[122]],
            [points_list[105], points_list[123]], [points_list[106], points_list[124]],
            [points_list[107], points_list[125]], [points_list[108], points_list[126]],
            [points_list[109], points_list[127]],
            [points_list[110], points_list[128]], [points_list[129], points_list[137]],
            [points_list[130], points_list[138]], [points_list[131], points_list[139]],
            [points_list[132], points_list[140]],
            [points_list[133], points_list[141]], [points_list[134], points_list[142]],
            [points_list[135], points_list[143]], [points_list[136], points_list[144]]]

# print(len(box_list))


def masktoLabel(targetImg, d, m ):

    oh, ow = features_map
    # print(oh, ow)
    targetlist = []
    dlist = []
    mlist = []
    for box in box_list:
        # h_min = round(int(box[0][1])/480*h)
        # w_min = round(int(box[0][0])/848*w)
        # h_max = round(int(box[1][1])/480*h)
        # w_max = round(int(box[1][0])/848*w)
        h = (int(box[0][1]) + int(box[1][1])) / 2
        w = (int(box[0][0]) + int(box[1][0])) / 2
        h = round(h / 480 * oh)
        w = round(w / 848 * ow)
        # print(h_min, w_min, h_max, w_max)
        newtarget = targetImg[:, h, w]
        targetlist.append(newtarget)
        newd = d[:, h, w]
        dlist.append(newd)
        newm = m[:, h, w]
        mlist.append(newm)
    # print(dlist)
    # print(targetlist)
    return targetlist, dlist, mlist
    # pass

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
    running_loss = []
    running_acc = []

    running_acc2 = []
    global best_correct
    for batch_idx, (data, target, filenames) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        target = torch.nn.functional.interpolate(target, size=features_map, mode='nearest')
        output = model(data).float()
        # loss = criterion(output, target)
        # output = torch.round(torch.sigmoid(output))
        output = torch.sigmoid(output)
        pred_cm(torch.Tensor.cpu(target[0]), torch.Tensor.cpu(output[0]))

    # now_correct = total_batch_acc
    # if best_correct < now_correct:
    #     best_correct = now_correct
    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     torch.save(best_model_wts, os.path.join(os.getcwd(), "mobilenetv2_fp16_34294rgbd_1248output.pth.tar"))
    #     print("New weight!")


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
title_name = 'yolic'
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





