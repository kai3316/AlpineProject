import os
import cv2
from collections import Counter
import numpy as np

# images_path = r"C:\Users\Kai\Downloads\data_semantics\training\semantic"
images_path = "/home/kai/Desktop/KITTI/training/semantic"
rgb_path = "/home/kai/Desktop/KITTI/training/semantic_rgb"

save_path = "/home/kai/Desktop/KITTI/training/semantic_rgb_224_224_2_2"
# resizedLabel_path = "/home/kai/Desktop/KITTI/training/semantic_14"
# os.makedirs(resizedLabel_path, exist_ok=True)
if not os.path.exists(save_path):
    os.makedirs(save_path)
all_images = os.listdir(images_path)
img_h = 224
img_w = 224
stepSize_h = 2  # 375
stepSize_w = 2  # 1242
windowSize = (stepSize_h, stepSize_w)

colorList = set()
color = []
print(int(img_h / stepSize_h), int(img_w / stepSize_w))
newImg = np.zeros((int(img_h / stepSize_h), int(img_w / stepSize_w), 3), np.uint8)  # (h,w,c)
color_dict = {7: [128, 64, 128], 8: [244, 35, 232], 11: [70, 70, 70], 12: [102, 102, 156], 13: [190, 153, 153],
              17: [153, 153, 153], 19: [250, 170, 30], 20: [220, 220, 0], 21: [107, 142, 35], 22: [152, 251, 152],
              23: [70, 130, 180], 24: [220, 20, 60], 25: [255, 0, 0], 26: [0, 0, 142], 27: [0, 0, 70], 28: [0, 60, 100],
              31: [0, 80, 100], 32: [0, 0, 230], 33: [119, 11, 32]}
index_dict = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
              27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


def detectColor(subImage, iy, ix):
    # print(subImage.shape)
    i = 0
    subColor = []
    for y in range(subImage.shape[0]):
        for x in range(subImage.shape[1]):
            temp = subImage[y][x]
            if not (
                    temp == 0 or temp == 1 or temp == 2 or temp == 3 or temp == 4 or temp == 5 or temp == 6 or temp == 9 or temp == 10 or temp == 14 or temp == 15 or temp == 16 or temp == 18 or temp == 29 or temp == 30):
                color.append(temp)
                subColor.append(temp)
                i = 1
    counterResult = Counter(subColor)
    # print(emptyImg[y][x])
    if len(counterResult) != 0:
        for item in counterResult:
            print(index_dict[item], counterResult[item])
        newImg[iy][ix][0] = color_dict[counterResult.most_common(1)[0][0]][2]
        newImg[iy][ix][1] = color_dict[counterResult.most_common(1)[0][0]][1]
        newImg[iy][ix][2] = color_dict[counterResult.most_common(1)[0][0]][0]
        # newImg[iy][ix] = counterResult.most_common(1)[0][0]
        # newImg[iy][ix][1] = counterResult.most_common(1)[0][0]
        # newImg[iy][ix][2] = counterResult.most_common(1)[0][0]


for path_image in all_images:
    fullPath = os.path.join(images_path, path_image)
    # rgbPath = os.path.join(rgb_path, path_image)
    print(fullPath)
    image = cv2.imread(fullPath, -1)
    # rgbimage = cv2.imread(rgbPath, -1)
    image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_NEAREST)  # (w,h)
    # rgbimage = cv2.resize(rgbimage, (16, 16), interpolation = cv2.INTER_NEAREST)  # (w,h)
    # print(image.shape)
    newImg = np.zeros((int(img_h / stepSize_h), int(img_w / stepSize_w), 3), np.uint8)  # (h,w,c)
    # cv2.imshow('emptyImg', emptyImg)
    # cv2.waitKey(0)

    # print(image.shape)
    for y in range(0, image.shape[0], stepSize_h):
        for x in range(0, image.shape[1], stepSize_w):
            subImage = image[y:y + windowSize[0], x:x + windowSize[1]]
            # print((y, y + windowSize[0]), (x, x + windowSize[1]))
            # print(subImage.shape)
            # print(y / stepSize_y, x / stepSize_x)
            detectColor(subImage, int(y / stepSize_h), int(x / stepSize_w))

            # print(subImage.shape)
            # cv2.rectangle(image, (x, y), (x + windowSize[1], y + windowSize[0]), (0, 0, 0), 1)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # exit(0)

    # print(Counter(color))
    # rgbImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test", newImg)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(save_path, path_image), newImg)
    # cv2.imwrite(os.path.join(resizedLabel_path, path_image), rgbimage)
