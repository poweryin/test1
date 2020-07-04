# coding:utf-8
import cv2
import numpy as np
from PIL import Image
import os
import re
# img1 = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/Arson016_x264/863.jpg') #.astype(np.int16)
# img2=cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/Arson016_x264/984.jpg')
'''
np.float32变为np.int32/int16也可以，
int32可以保存-2147483648~214748364
int16可以保存-32768~32767，
int16类似于CV_16SC1
int32类似于CV_32SC1'''

mask_path="/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/Arson16_x264/"
mask_=os.listdir(mask_path)
mask_.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
i=-1
for mask_id in mask_:
    id_path=os.path.join(mask_path,mask_id)
    id2_path=os.path.join(mask_path,mask_[i])
    img1 = np.array(Image.open(id_path).convert('L'))
    img2 = np.array(Image.open('/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/Arson16_x264/1078.jpg').convert('L'))
    err = cv2.absdiff(img1,img2)     #差值的绝对值
    err1 = np.abs(img1 - img2)               #差值
    errdiff =err-err1
    i-=1
    cv2.imwrite("/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/diff/{}".format(mask_id), errdiff)