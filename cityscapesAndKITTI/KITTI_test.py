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
import cv2
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import os
from modelKITTI import mobilenet_v2

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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




class DatasetKITTI(torch.utils.data.Dataset):
    def __init__(self, KITTI_data_path, KITTI_meta_path):
        self.img_dir = KITTI_data_path + "/training/image_2/"
        self.label_dir = KITTI_meta_path + "/label_imgs/"

        self.imgLabel_h = 75
        self.imgLabel_w = 207

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
        img_id = example["img_id"]
        img_path = example["img_path"]
        img = cv2.imread(img_path, -1)  # (shape: (1024, 2048, 3))
        # resize img without interpolation (want the image to still match
        # label_img, which we resize below):
        img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                         interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024, 3))

        label_img_path = example["label_img_path"]
        label_img = cv2.imread(label_img_path, -1)  # (shape: (75, 207,3 ))
        label_img, _, _ = cv2.split(label_img)  # (shape: (75, 207))
        # resize label_img without interpolation (want the resulting image to
        # still only contain pixel values corresponding to an object class):
        label_img = cv2.resize(label_img, (self.imgLabel_w, self.imgLabel_h),
                               interpolation=cv2.INTER_NEAREST)  # (shape: (512, 1024))

        # flip the img and the label with 0.5 probability:
        # flip = np.random.randint(low=0, high=2)
        # if flip == 1:
        #     img = cv2.flip(img, 1)
        #     label_img = cv2.flip(label_img, 1)
        #
        # ########################################################################
        # # randomly scale the img and the label:
        # ########################################################################
        # scale = np.random.uniform(low=0.7, high=1.5)
        # new_img_h = int(scale * self.new_img_h)
        # new_img_w = int(scale * self.new_img_w)
        #
        # imgLabel_h = int(scale * self.imgLabel_h)
        # imgLabel_w = int(scale * self.imgLabel_w)
        # # resize img without interpolation (want the image to still match
        # # label_img, which we resize below):
        # img = cv2.resize(img, (new_img_w, new_img_h),
        #                  interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w, 3))
        #
        # # resize label_img without interpolation (want the resulting image to
        # # still only contain pixel values corresponding to an object class):
        # label_img = cv2.resize(label_img, (imgLabel_w, imgLabel_h),
        #                        interpolation=cv2.INTER_NEAREST)  # (shape: (new_img_h, new_img_w))
        ########################################################################

        # # # # # # # # debug visualization START
        # print(scale)
        # print(self.new_img_h)
        # print(self.new_img_w)

        # cv2.imshow("test", img)
        # cv2.waitKey(0)

        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        ########################################################################
        # select a 256x256 random crop from the img and label:
        ########################################################################
        # start_x = np.random.randint(low=0, high=(new_img_w - 256))
        # end_x = start_x + 256
        # start_y = np.random.randint(low=0, high=(new_img_h - 256))
        # end_y = start_y + 256
        #
        # img = img[start_y:end_y, start_x:end_x]  # (shape: (256, 256, 3))
        # label_img = label_img[start_y:end_y, start_x:end_x]  # (shape: (256, 256))
        ########################################################################

        # # # # # # # # debug visualization START
        # print (img.shape)
        # print (label_img.shape)
        #
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        #
        # cv2.imshow("test", label_img)
        # cv2.waitKey(0)
        # # # # # # # # debug visualization END

        # normalize the img (with the mean and std for the pretrained ResNet):
        img = img / 255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img / np.array([0.229, 0.224, 0.225])  # (shape: (256, 256, 3))
        img = np.transpose(img, (2, 0, 1))  # (shape: (3, 256, 256))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img)  # (shape: (3, 256, 256))
        label_img = torch.from_numpy(label_img)  # (shape: (75, 207))
        # print(label_img.shape)
        return img, label_img, img_id

    def __len__(self):
        return self.num_examples


train_dataset = DatasetKITTI(KITTI_data_path="/home/kai/Desktop/KITTI",
                             KITTI_meta_path="/home/kai/Desktop/KITTI/meta")

x_train, x_test = train_test_split(train_dataset, test_size=0.2, random_state=2)

train_loader = torch.utils.data.DataLoader(dataset=x_train,
                                           batch_size=args.batch_size,
                                           shuffle=True, num_workers=5)

test_loader = torch.utils.data.DataLoader(x_test,
                                          batch_size=args.test_batch,
                                          shuffle=False, num_workers=5)

model_id = "2"
model = mobilenet_v2(model_id = model_id, project_dir="/home/kai/Desktop/KITTI")
# save_name = 'mobilenet_v2_40760'
checkpoint = torch.load("/home/kai/Desktop/KITTI/training_logs/model_2/checkpoints/model_2_epoch_1000.pth")
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
learning_rate = 0.0001
optimizer = optim.Adam(params, lr=learning_rate)

with open("/home/kai/Desktop/KITTI/meta/class_weights.pkl", "rb") as file:  # (needed for python3)
    class_weights = np.array(pickle.load(file))
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)


def train(epoch, model, loss_fn):
    model.train()
    batch_losses = []
    for batch_idx, (data, target, img_id) in enumerate(train_loader):
        if args.cuda:
            data, target = Variable(data).cuda(), Variable(target.type(torch.LongTensor)).cuda()
        optimizer.zero_grad()
        # with autocast():
        output = model(data)
        # print(data.shape, output.shape)
        # print(output.shape, target.shape)
        # print(target.shape)
        # exit(0)
        loss = loss_fn(output, target)
        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)
        # print(loss)
        # loss = F.softmax(output, target)
        # loss = F.cross_entropy(output, target)
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


def train_evaluate():
    model.eval()
    batch_losses = []
    for step, (imgs, label_imgs, img_ids) in enumerate(test_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

            outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
            outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)
            for i in range(pred_label_imgs.shape[0]):
                pred_label_img = pred_label_imgs[i] # (shape: (img_h, img_w))
                img_id = img_ids[i]
                img = imgs[i] # (shape: (3, img_h, img_w))

                img = img.data.cpu().numpy()
                img = np.transpose(img, (1, 2, 0)) # (shape: (img_h, img_w, 3))
                # img = img*np.array([0.229, 0.224, 0.225])
                # img = img + np.array([0.485, 0.456, 0.406])
                img = img*255.0
                img = img.astype(np.uint8)
                print(pred_label_img, pred_label_img.shape)

                pred_label_img_color = label_img_to_color(pred_label_img)
                # overlayed_img = 0.35*img + 0.65*pred_label_img_color
                # overlayed_img = overlayed_img.astype(np.uint8)

                # img_h = overlayed_img.shape[0]
                # img_w = overlayed_img.shape[1]
                # print(model.results_dir + "/" + img_id)
                # print(pred_label_img_color.shape)
                pred_label_img_color = cv2.cvtColor(pred_label_img_color.astype(np.float32), cv2.COLOR_BGR2RGB)
                cv2.imwrite(model.results_dir + "/" + img_id , img)
                cv2.imwrite(model.results_dir + "/pred_" + img_id , pred_label_img_color)
                # cv2.imwrite(model.model_dir + "/" + img_id + "_overlayed.png", overlayed_img)

            # # compute the loss:
            # loss = loss_fn(outputs, label_imgs)
            # loss_value = loss.data.cpu().numpy()
            # batch_losses.append(loss_value)

    # epoch_loss = np.mean(batch_losses)
    # epoch_losses_val.append(epoch_loss)
    # with open("%s/epoch_losses_val.pkl" % model.model_dir, "wb") as file:
    #     pickle.dump(epoch_losses_val, file)
    # print ("val loss: %g" % epoch_loss)
    # plt.figure(1)
    # plt.plot(epoch_losses_val, "k^")
    # plt.plot(epoch_losses_val, "k")
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.title("val loss per epoch")
    # plt.savefig("%s/epoch_losses_val.png" % model.model_dir)
    # plt.close(1)
    #
    # # save the model weights to disk:
    # checkpoint_path = model.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    # torch.save(model.state_dict(), checkpoint_path)

def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color
if __name__ == '__main__':
    epoch_losses_train = []
    epoch_losses_val = []
    train_evaluate()


