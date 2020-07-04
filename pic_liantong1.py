# coding:utf-8
import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt
# 利用opencv获取最大连通区域,并画出bounding box
def largestConnectComponent(bw_img):
    labeled_img, num = skimage.measure.label(bw_img, neighbors=4, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(0, num):
        if np.sum(labeled_img == 1) > max_num:
            max_num = np.sum(labeled_img == 1)
            max_label = i
    mcr = (labeled_img == max_label)
    return mcr

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(mcr)
for region in regionprops(labeled_img):
    # skip small images
    if region.area < 50:
        continue
    # print(regionprops(labeled_img)[max_label])
    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

















mask= cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/Arson016_x264/950.jpg')
# 进行图像的二值化处理
max_value = np.max(mask)
#print(max_value)
ret, ee = cv2.threshold(mask, 0.15*max_value, max_value, cv2.THRESH_BINARY)