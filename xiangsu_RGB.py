# coding:utf-8
import cv2 as cv
import numpy as np
import os
import re
import scipy.io as scio
import matplotlib.pyplot as plt
def access_pixels(frame):
    print(frame.shape)  # shape内包含三个元素：按顺序为高、宽、通道数
    height = frame.shape[0]
    weight = frame.shape[1]
    channels = frame.shape[2]
    print("weight : %s, height : %s, channel : %s" % (weight, height, channels))

    for row in range(height):  # 遍历高
        for col in range(weight):  # 遍历宽
            for c in range(channels):  # 便利通道
                pv = frame[row, col, c]
                # frame[row, col, c] = 255 - pv  # 全部像素取反，实现一个反向效果
    cv.imshow("fanxiang", frame)


image = "/home/z840/dataset/UCF_Crimes/test/000001.jpg"
src = cv.imread(image)
cv.imshow("Picture", src)
access_pixels(src)
cv.waitKey(0)