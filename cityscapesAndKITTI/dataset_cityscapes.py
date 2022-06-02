import cv2
import os
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from collections import Counter

color_dict = {7: [128, 64, 128], 8: [244, 35, 232], 11: [70, 70, 70], 12: [102, 102, 156], 13: [190, 153, 153],
              17: [153, 153, 153], 19: [250, 170, 30], 20: [220, 220, 0], 21: [107, 142, 35], 22: [152, 251, 152],
              23: [70, 130, 180], 24: [220, 20, 60], 25: [255, 0, 0], 26: [0, 0, 142], 27: [0, 0, 70], 28: [0, 60, 100],
              31: [0, 80, 100], 32: [0, 0, 230], 33: [119, 11, 32]}
index_dict = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
              27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


class dataset_Cityscapes(torch.utils.data.Dataset):
    def __init__(self, imgspath, imgslist, annotationpath, img_w=224, img_h=224, cell_size_h=2, cell_size_w=2,
                 num_classes=20, transforms=None):
        self.imagesList = imgslist
        self.imagesPath = imgspath
        self.transform = transforms
        self.annotationPath = annotationpath
        self.img_w = img_w
        self.img_h = img_h
        self.cell_size_h = cell_size_h
        self.cell_size_w = cell_size_w
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imagesList)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imagesPath, self.imagesList[index])
        labeledImgPath = os.path.join(self.annotationPath, self.imagesList[index])
        colorImage = cv2.imread(imgPath, -1)
        label_image = cv2.imread(labeledImgPath, -1)
        colorImage = cv2.resize(colorImage, (self.img_w, self.img_h))  # (w,h)
        label_image = cv2.resize(label_image, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)  # (w,h)
        cellLabelImage = np.zeros(
            (self.num_classes, int(self.img_h / self.cell_size_h), int(self.img_w / self.cell_size_w)),
            np.uint8)  # (h,w,c)
        for y in range(0, label_image.shape[0], self.cell_size_h):
            for x in range(0, label_image.shape[1], self.cell_size_w):
                subImage = label_image[y:y + self.cell_size_h, x:x + self.cell_size_w]
                self.regionSearch(subImage, int(y / self.cell_size_h), int(x / self.cell_size_w), cellLabelImage)

        if self.transform is not None:
            colorImage = self.transform(colorImage)

        return colorImage, cellLabelImage

    @staticmethod
    def regionSearch(subImage, iy, ix, newImg):
        # print(subImage.shape)
        subColor = []
        for y in range(subImage.shape[0]):
            for x in range(subImage.shape[1]):
                temp = subImage[y][x]
                if not (
                        temp == 0 or temp == 1 or temp == 2 or temp == 3 or temp == 4 or temp == 5 or temp == 6 or temp == 9 or temp == 10 or temp == 14 or temp == 15 or temp == 16 or temp == 18 or temp == 29 or temp == 30):
                    subColor.append(temp)
        counterResult = Counter(subColor)
        # print(emptyImg[y][x])
        if len(counterResult) != 0:
            for item in counterResult:
                # print(item, index_dict[item])
                newImg[index_dict[item]][iy][ix] = 1
        else:
            newImg[19][iy][ix] = 1


if __name__ == '__main__':
    rgb_dir = '/home/kai/Desktop/KITTI/training/image_2'
    label_dir = '/home/kai/Desktop/KITTI/training/semantic'
    alist = os.listdir(rgb_dir)
    x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)
    train = dataset_Cityscapes(rgb_dir, x_test, label_dir)
    color_image, newLabel_image = train.__getitem__(0)
    # cv2.imshow('color_image', color_image)
    # cv2.waitKey(0)
