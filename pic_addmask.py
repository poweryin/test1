#coding:utf-8
import os
import cv2
import numpy as np
import re
def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    file_path=os.listdir(images_path)
    file_path.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for img_item in file_path:
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-4]+'.jpg')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE,)  # 将彩色mask以二值图像形式读取
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, img_item), masked)
images_path = '/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/img/Explosion002_x264'
masks_path = '/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/Explosion002_x264'
masked_path = '/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/mask_/Explosion002_x264'
add_mask2image_binary(images_path, masks_path, masked_path)

