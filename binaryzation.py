import cv2
import numpy as np
im = cv2.imread(r"C:\Users\Su\Desktop\1_Depth.png")
im = cv2.resize(im,(848,480))
grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow('test',im)

#二值化处理，低于阈值的像素点灰度值置为0；高于阈值的值置为参数3
ret,thresh1 = cv2.threshold(grayImage,50,255,cv2.THRESH_BINARY)
cv2.imshow('BINARY',thresh1)
thresh1 = thresh1[240:480, 0:848]
cv2.imshow('BINARY1',thresh1)
print(len(thresh1[thresh1==0]))

cv2.waitKey(0)