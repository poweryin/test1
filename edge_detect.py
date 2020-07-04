# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:23:32 2018
@author: Leon
内容：
对图片进行边缘检测；
添加滑动条，可自由调整阈值上下限。
"""

import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow('Canny', 0)
# 创建滑动条
cv2.createTrackbar('minval', 'Canny', 0, 255, nothing)
cv2.createTrackbar('maxval', 'Canny', 0, 255, nothing)

img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_test_out_30/bg/Arson016_x264/1607.jpg', 0)

# 高斯滤波去噪
img = cv2.GaussianBlur(img, (3, 3), 0)
edges = img

k = 0
while (1):

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    # 读取滑动条数值
    minval = cv2.getTrackbarPos('minval', 'Canny')
    maxval = cv2.getTrackbarPos('maxval', 'Canny')
    edges = cv2.Canny(img, minval, maxval)

    # 拼接原图与边缘监测结果图
    img_2 = np.hstack((img, edges))
    cv2.imshow('Canny', img_2)

cv2.destroyAllWindows()