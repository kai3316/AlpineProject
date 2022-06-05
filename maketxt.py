# 将数据集划分，并且构建2个TXT：train.txt，test.txt
import os

from sklearn.model_selection import train_test_split


def maketxt(data_dir, train_txt, test_txt):
    alist = os.listdir(data_dir)
    x_train, x_test = train_test_split(alist, test_size=0.3, random_state=2)
    # 写入train.txt
    with open(train_txt, 'w') as f:
        for i in x_train:
            f.write(i[:-4] + '\n')
    # 写入test.txt
    with open(test_txt, 'w') as f:
        for i in x_test:
            f.write(i[:-4] + '\n')


data_dir = '/home/kai/Desktop/RGB/'
train_txt = '/home/kai/Desktop/train.txt'
test_txt = '/home/kai/Desktop/test.txt'
maketxt(data_dir, train_txt, test_txt)
