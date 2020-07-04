# coding:utf-8
import cv2
import numpy as np
from PIL import Image
import os
import re
#图像减法 - 血液流动
import cv2
# # ori1 = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/bg/Arson041_x264/001765.jpg')
# ori1 = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/bg/Arson016_x264/001000.jpg')
# ori1 = cv2.cvtColor(ori1,cv2.COLOR_RGB2GRAY)
# # ori2 = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/Arson041_x264/1765.jpg')
# ori2 = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/Arson016_x264/1000.jpg')
# ori2 = cv2.cvtColor(ori2,cv2.COLOR_RGB2GRAY)
# cv2.imshow('minus1',ori1)
# cv2.imshow('minus2',ori2)
#
# # cv2.waitKey()
# city3 = ori2 - ori1
# city3[city3 <= 150] = 255
# # city3[city3 > 160] = 0
# cv2.imshow('city',city3)
# i=1155
# cv2.imwrite("/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/{}.jpg".format(i), city3)
# cv2.waitKey()

#!/usr/bin/env python

import cv2
import numpy as np
# ph1 = "/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/img/Arson016_x264/863.jpg"
# ph2 = "/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/img/Arson016_x264/1000.jpg"
id_path= "/home/z840/dataset/UCF_Crimes/crop_out_30/simi2/Arson016_x264/876.jpg"
id2_path= "/home/z840/dataset/UCF_Crimes/crop_out_30/simi2/Arson016_x264/1117.jpg"
# tmp=ph2.split('/')[-1].split('.')[0]
# threshod= 150
#
# s1 = cv2.imread(ph1,1)
# s2 = cv2.imread(ph2,1)

img1 = np.array(Image.open(id_path))  #.convert('L'))
img2 = np.array(Image.open(id2_path)) #.convert('L'))
err = cv2.absdiff(img1, img2)  # 差值的绝对值
err1 = np.abs(img1 - img2)  # 差值
errdiff = err - err1
mask_id=876
cv2.imwrite("/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/diff/{}.jpg".format(mask_id), errdiff)


# emptyimg = np.zeros(s1.shape,np.uint8)


# def pic_sub(dest,s1,s2):
#     for x in range(dest.shape[0]):
#         for y in range(dest.shape[1]):
#             if(s2[x,y] > s1[x,y]):
#                 dest[x,y] = s2[x,y] - s1[x,y]
#             else:
#                 dest[x,y] = s1[x,y] - s2[x,y]
#
#             if(dest[x,y] < threshod):
#                 dest[x,y] = 0
#             else:
#                 dest[x,y] = 255



# pic_sub(emptyimg,s1,s2)

cv2.namedWindow("s1")
cv2.namedWindow("s2")
cv2.namedWindow("result")

cv2.imshow("s1",img1)
cv2.imshow("s2",img2)
cv2.imshow("result",errdiff)
i=1100
# cv2.imwrite("/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/{}_.jpg".format(i),dest)

cv2.waitKey(0)
cv2.destroyAllWindows()