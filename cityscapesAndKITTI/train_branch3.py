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
import pickle
from torch.autograd import Variable
import time
import copy
import seaborn as sns
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import os
from YOLIC import YOLIC_MobileNetV2
from collections import Counter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from model.deeplabv3 import DeepLabV3


parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=16, metavar='N',
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
parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=True, metavar='N',
                    help='resume from the last weights')







torch.cuda.empty_cache()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




def detectColor(subImage, iy, ix, newImg):
    # print(subImage.shape)
    i = 0
    subColor = []
    for y in range(subImage.shape[0]):
        for x in range(subImage.shape[1]):
            temp = subImage[y][x]
            subColor.append(temp.item())
    counterResult = Counter(subColor)
    # print(counterResult)
    # print(emptyImg[y][x])

    # print(counterResult.most_common(1)[0][0])
    # newImg[iy][ix][0] = color_dict[counterResult.most_common(1)[0][0]][2]
    # newImg[iy][ix][1] = color_dict[counterResult.most_common(1)[0][0]][1]
    # newImg[iy][ix][2] = color_dict[counterResult.most_common(1)[0][0]][0]
    for counterResultKey in counterResult:
        if(counterResult[counterResultKey]) > 10:
            # print(counterResultKey)
            newImg[counterResultKey][iy][ix] = 1

    # print(counterResult.most_common(1)[0][0])

    # newImg[iy][ix] = counterResult.most_common(1)[0][0]


def sm2cwsm(label_img, stepSize_y, stepSize_x):
    newImg = np.zeros((20, int(label_img.shape[0] / stepSize_y), int(label_img.shape[1] / stepSize_x)), np.uint8)
    subImageTensor = np.zeros(
        (int(label_img.shape[0] / stepSize_y) * int(label_img.shape[1] / stepSize_x), stepSize_y, stepSize_x),
        dtype=np.uint8)
    # print(image.shape)
    i = 0
    for y in range(0, label_img.shape[0], stepSize_y):
        for x in range(0, label_img.shape[1], stepSize_x):
            subImage = label_img[y:y + stepSize_y, x:x + stepSize_x]
            subImageTensor[i] = subImage
            i += 1
            detectColor(subImage, int(y / stepSize_y), int(x / stepSize_x), newImg)
    subImageTensor = torch.from_numpy(subImageTensor)
    return newImg, subImageTensor


def Tsm2cwsm(label_img, stepSize_y, stepSize_x):
    newImg = np.zeros(
        (label_img.size()[0], 20, int(label_img.size()[1] / stepSize_y), int(label_img.size()[2] / stepSize_x)),
        np.uint8)
    # subImageTensor = torch.zeros((label_img.size()[0],
    #                               int(label_img.size()[1] / stepSize_y) * int(label_img.size()[2] / stepSize_x),
    #                               stepSize_y, stepSize_x), dtype=torch.uint8)
    # print(subImageTensor.shape)
    # print(image.shape)
    for each in range(label_img.size()[0]):
        eachImg = label_img[each]
        # print(eachImg.shape)
        i = 0
        for y in range(0, eachImg.shape[0], stepSize_y):
            for x in range(0, eachImg.shape[1], stepSize_x):
                subImage = eachImg[y:y + stepSize_y, x:x + stepSize_x]
                # print(subImage.shape)
                # subImageTensor[each][i] = subImage
                i += 1
                detectColor(subImage, int(y / stepSize_y), int(x / stepSize_x), newImg[each])

    return newImg


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, KITTI_data_path, KITTI_meta_path):
        self.img_dir = KITTI_data_path + "/training/image_2/"
        self.label_dir = KITTI_meta_path + "/label_imgs_375_1242/"

        self.imgLabelForBranch1_h = 75
        self.imgLabelForBranch1_w = 207

        self.imgLabelForBranch2_h = 15
        self.imgLabelForBranch2_w = 23

        self.new_img_h = 375
        self.new_img_w = 1242

        self.examples = []
        file_names = os.listdir(self.img_dir)

        for file_name in file_names:
            img_id = file_name
            label_img_path = self.label_dir + img_id

            example = {}
            example["img_path"] = self.img_dir + img_id
            example["label_img_path"] = label_img_path
            example["img_id"] = img_id
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)  # (shape: (375, 1242,3 ))
        label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                               interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        new_labelForBranch1, subImageTensor = sm2cwsm(label_img, self.imgLabelForBranch1_h, self.imgLabelForBranch1_w)
        new_labelForBranch2 = Tsm2cwsm(subImageTensor, self.imgLabelForBranch2_h,
                                                        self.imgLabelForBranch2_w)
        img = img / 255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img)  # (shape: (75, 207))
        # print(label_img.shape)
        new_labelForBranch1 = torch.from_numpy(new_labelForBranch1)
        new_labelForBranch2 = torch.from_numpy(new_labelForBranch2)
        # print(new_labelForBranch1.shape, new_labelForBranch2.shape, subImageTensor2.shape)
        #  torch.Size([20, 5, 6]) torch.Size([30, 20, 5, 9]) torch.Size([30, 45, 15, 23])
        return img, new_labelForBranch1, new_labelForBranch2, label_img

    def __len__(self):
        return self.num_examples


train_dataset = DatasetTrain(KITTI_data_path="/home/kai/Desktop/KITTI",
                             KITTI_meta_path="/home/kai/Desktop/KITTI/meta")

x_train, x_test = train_test_split(train_dataset, test_size=0.1, random_state=2)

train_loader = torch.utils.data.DataLoader(dataset=x_train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=10)

test_loader = torch.utils.data.DataLoader(x_test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=10)

model = DeepLabV3(model_id=6, project_dir="./")
model_id = "6"
# save_name = 'mobilenet_v2_40760'
checkpoint = torch.load("/home/kai/Desktop/AlpineProject/cityscapesAndKITTI/training_logs/model_5/checkpoints/model_5_epoch_60.pth")
model.load_state_dict(checkpoint)
model1 = model.parameters()
if args.cuda:
    model.cuda()


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


params = add_weight_decay(model, l2_value=0.0001)
learning_rate = 0.00001
optimizer = optim.Adam(params, lr=learning_rate)


with open("/home/kai/Desktop/KITTI/meta/class_weights.pkl", "rb") as file:  # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.MultiLabelSoftMarginLoss()
loss_ce_fn = nn.CrossEntropyLoss(weight=class_weights)


# def batch_processing(Feature_branch1):
#     """Feature_branch1: (2,64, 375, 1242), step size: (75, 207)
#     """
#     num_subImage = Feature_branch1.shape[2] // 75 * Feature_branch1.shape[3] // 207
#     print(num_subImage)
#     input_tensor = torch.zeros(Feature_branch1.size()[0], num_subImage, 64, 75, 207)
#     # print(input_tensor.shape)
#     for batch in range(Feature_branch1.size()[0]):
#         for i in range(5):
#             for j in range(6):
#                 input_tensor[batch, i * 6 + j, :, :, :] = Feature_branch1[batch, :, i * 75:i * 75 + 75,
#                                                           j * 207:j * 207 + 207]
#
#     input_tensor = input_tensor.view(input_tensor.size()[0] * num_subImage, 64, 75, 207)
#     return input_tensor


def train(epoch, model, loss_fn, ce_loss_fn):
    model.train()
    batch_losses = []

    for batch_idx, (data, new_labelForBranch1, new_labelForBranch2, LabelImage) in enumerate(train_loader):
        if args.cuda:
            data, new_labelForBranch1, new_labelForBranch2, LabelImage = data.cuda(), Variable(
                new_labelForBranch1.type(torch.LongTensor)).cuda(), Variable(
                new_labelForBranch2.type(torch.LongTensor)).cuda(), Variable(
                LabelImage.type(torch.LongTensor)).cuda()

        optimizer.zero_grad()

        # with autocast():


        # print(out_branch2.shape)
        branch3Setting = {"x": np.random.randint(0, 25), "y": np.random.randint(0, 54)}
        point_x = branch3Setting["x"] * 15
        point_y = branch3Setting["y"] * 23
        # print(x.shape)
        # print(point_x, point_y)
        interesting_region = data[:, :, point_x: point_x + 15, point_y: point_y + 23]
        out_branch3 = model(interesting_region)
        subLabel = LabelImage[:, branch3Setting["x"] * 15: branch3Setting["x"] * 15 + 15,
                   branch3Setting["y"] * 23: branch3Setting["y"] * 23 + 23]
        # print(LabelImage.shape, subLabel.shape) # torch.Size([2, 375, 1242]) torch.Size([2, 15, 23])
        # print(out_branch3.shape)  # torch.Size([2, 20, 15, 23])
        branch3_loss = ce_loss_fn(out_branch3, subLabel)

        loss = branch3_loss
        # loss = branch1_loss + branch2_loss + branch3_loss
        # print(branch1_loss, branch2_loss, branch3_loss)
        # inputForBatch2 = batch_processing(Feature_branch1)
        # print(inputForBatch2.shape)
        # torch.Size([20, 5, 6])
        # torch.Size([30, 20, 5, 9])
        # torch.Size([30, 45, 15, 23])
        # output_branch2 = model(inputForBatch2, "branch2")
        # print(output_branch2.shape)

        # output = model(data)
        # print(data.shape, output.shape)
        # print(output.shape, target.shape)
        # print(target.shape)
        # exit(0)
        # loss = loss_fn(output, output)
        # print(loss)
        loss_value = loss.data.cpu().numpy()  # torch.Size([batch_size, 20, 75, 207]) torch.Size([batch_size, 75, 207])

        batch_losses.append(loss_value)

        # print(loss)
        # loss = F.softmax(output, target)
        # loss = F.cross_entropy(output, target)
        # loss.backward()

        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.step()

        # pred = output.data.max(1, keepdim=True)[1]

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % model.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % model.model_dir)
    plt.close(1)




def train_evaluate(loss_fn, ce_loss_fn):
    model.eval()
    batch_losses = []

    for batch_idx, (imgs, new_labelForBranch1, new_labelForBranch2, LabelImage) in enumerate(test_loader):
        with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs, new_labelForBranch1, new_labelForBranch2, LabelImage = imgs.cuda(), Variable(
                new_labelForBranch1.type(torch.LongTensor)).cuda(), Variable(
                new_labelForBranch2.type(torch.LongTensor)).cuda(), Variable(
                LabelImage.type(torch.LongTensor)).cuda()



            # print(out_branch2.shape)
            branch3Setting = {"x": np.random.randint(0, 25), "y": np.random.randint(0, 54)}
            point_x = branch3Setting["x"] * 15
            point_y = branch3Setting["y"] * 23
            # print(x.shape)
            # print(point_x, point_y)
            interesting_region = imgs[:, :, point_x: point_x + 15, point_y: point_y + 23]
            out_branch3 = model(interesting_region)
            subLabel = LabelImage[:, branch3Setting["x"] * 15: branch3Setting["x"] * 15 + 15,
                       branch3Setting["y"] * 23: branch3Setting["y"] * 23 + 23]
            # print(LabelImage.shape, subLabel.shape) # torch.Size([2, 375, 1242]) torch.Size([2, 15, 23])
            # print(out_branch3.shape)  # torch.Size([2, 20, 15, 23])
            branch3_loss = ce_loss_fn(out_branch3, subLabel)
            loss = branch3_loss
            # print(branch1_loss, branch2_loss, branch3_loss)

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)


    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % model.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % model.model_dir)
    plt.close(1)


    # save the model weights to disk:
    checkpoint_path = model.checkpoints_dir + "/model_" + model_id + "_epoch_" + str(epoch + 1) + ".pth"
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    epoch_losses_train = []

    epoch_losses_val = []

    for epoch in range(args.epochs):
        print("###########################")
        print("######## NEW EPOCH ########")
        print("###########################")
        print("epoch: %d/%d" % (epoch + 1, args.epochs))
        train(epoch, model, loss_fn=loss_fn, ce_loss_fn=loss_ce_fn)
        train_evaluate(loss_fn, ce_loss_fn=loss_ce_fn)
