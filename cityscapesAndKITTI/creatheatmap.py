import os
import cv2
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# images_path = r"C:\Users\Kai\Downloads\data_semantics\training\semantic"
images_path = "/home/kai/Desktop/KITTI/training/semantic"
save_path = "/home/kai/Desktop/KITTI/training/semantic_heatmap"
if not os.path.exists(save_path):
    os.makedirs(save_path)
all_images = os.listdir(images_path)
# stepSize_y = 5  # 375
# stepSize_x = 6  # 1242
windowSize = (75, 207)
stepSize_y = windowSize[0]
stepSize_x = windowSize[1]

image_height = 375
image_width = 1242

colorList = set()
color = []
heatmap = np.zeros((int(image_height/stepSize_y), int(image_width/stepSize_x)), np.uint8)
print(heatmap.shape)
color_dict = {7: [128, 64, 128], 8: [244, 35, 232], 11: [70, 70, 70], 12: [102, 102, 156], 13: [190, 153, 153], 17: [153, 153, 153], 19: [250, 170, 30], 20: [220, 220,  0], 21: [107, 142, 35], 22: [152, 251, 152], 23: [70, 130, 180], 24: [220, 20, 60], 25: [255,  0,  0], 26: [0,  0, 142], 27: [0,  0, 70], 28: [0, 60, 100], 31: [0, 80, 100], 32: [0,  0, 230], 33: [119, 11, 32]}
frequency = np.zeros((20, 1), np.uint8)

def detectColor(subImage, iy, ix):
    # print(subImage.shape)
    subColor = []
    for y in range(subImage.shape[0]):
        for x in range(subImage.shape[1]):
            temp = subImage[y][x]
            if not (
                    temp == 0 or temp == 1 or temp == 2 or temp == 3 or temp == 4 or temp == 5 or temp == 6 or temp == 9 or temp == 10 or temp == 14 or temp == 15 or temp == 16 or temp == 18 or temp == 29 or temp == 30):
                color.append(temp)
                subColor.append(temp)
    counterResult = Counter(subColor)
       # print(emptyImg[y][x])
    if len(counterResult) != 0:
        frequency[len(counterResult)] += 1
        # print(counterResult.most_common(1)[0][0])
        # newImg[iy][ix][0] = color_dict[counterResult.most_common(1)[0][0]][2]
        # newImg[iy][ix][1] = color_dict[counterResult.most_common(1)[0][0]][1]
        # newImg[iy][ix][2] = color_dict[counterResult.most_common(1)[0][0]][0]
        heatmap[iy][ix] = len(counterResult)
    if len(counterResult) >=10:
        print(counterResult)
        cv2.imshow('Image', subImage)
        cv2.waitKey(0)


for path_image in all_images:
    fullPath = os.path.join(images_path, path_image)
    print(fullPath)
    image = cv2.imread(fullPath, -1)

    image = cv2.resize(image, (image_width, image_height), interpolation= cv2.INTER_NEAREST)
    # print(image.shape)
    # cv2.imshow('emptyImg', emptyImg)
    # cv2.waitKey(0)

    # print(image.shape)
    for y in range(0, image.shape[0], stepSize_y):
        for x in range(0, image.shape[1], stepSize_x):
            subImage = image[y:y + windowSize[0], x:x + windowSize[1]]
            # print((y, y + windowSize[0]), (x, x + windowSize[1]))
            # print(subImage.shape)
            # print(y / stepSize_y, x / stepSize_x)
            detectColor(subImage, int(y/stepSize_y), int(x/stepSize_x))

            # print(subImage.shape)
            # cv2.rectangle(image, (x, y), (x + windowSize[1], y + windowSize[0]), (0, 0, 0), 1)
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    # exit(0)

    ax = sns.heatmap(heatmap, linewidth=0.7)
    plt.show()
print(frequency)
    # exit(0)
    # cv2.imshow('Image', heatmap)
    # cv2.waitKey(0)
    # exit(0)

    # print(Counter(color))
    # rgbImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test", newImg)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(save_path, path_image), newImg)

