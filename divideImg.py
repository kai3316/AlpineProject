# 将数据集划分，并且按照比例划分训练集和测试
import os

from sklearn.model_selection import train_test_split


def divideImg(data_dir, train_dir, test_dir):
    alist = os.listdir(data_dir)
    x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)
    # copu all the train images into train_dir
    for i in x_train:
        src = os.path.join(data_dir, i)
        dst = os.path.join(train_dir, i)
        os.system('cp ' + src + ' ' + dst)
    # copy all the test images into test_dir
    for i in x_test:
        src = os.path.join(data_dir, i)
        dst = os.path.join(test_dir, i)
        os.system('cp ' + src + ' ' + dst)


data_dir = '/home/kai/Desktop/data_noflip/RGB_noflip/'
train_dir = '/home/kai/Desktop/data_noflip/trainRGB/'
test_dir = '/home/kai/Desktop/data_noflip/testRGB/'
# create train_dir and test_dir
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
divideImg(data_dir, train_dir, test_dir)
