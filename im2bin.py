# coding:utf-8
# 图片二值化
from PIL import Image
import os
import re
import cv2
import numpy as np
input_path="/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/silance/"
bin_path="/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bin/"
All_Folder = os.listdir(input_path)
All_Folder.sort()
# All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
for video_name in All_Folder:
    filepath = os.path.join(input_path, video_name)
    # filepath_ = os.path.join(res_path, video_name)
    # diffpath_ = os.path.join(result_path, video_name)
    # if not os.path.exists(diffpath_):
    #     os.makedirs(diffpath_)
    # read binary data
    pic_folder = os.listdir(filepath)
    lenth = len(pic_folder)
    pic_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    # pic_folder.sort(key=lambda x: list(map(int, x.split('.')[0].split('_'))))
    for id in pic_folder:
        pic_path=os.path.join(filepath,id)
        id_name=id.split('_')[0]
        # img = Image.open(pic_path)
        img=cv2.imread(pic_path)
        crop_size=(320, 240)
        img_ = cv2.resize(img, crop_size, interpolation=cv2.INTER_CUBIC)
        # img = img / 255
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img_ = img_/255
        img_ [img_<0.5] = 0
        img_[img_>=0.5] = 255

        # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
        # Img = img.convert('L')
        # Img.save("test1.jpg")
        # cv2.imshow("ori",Img)
        # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
        # threshold = 128
        #
        # table = []
        # for i in range(256):
        #     if i < threshold:
        #         table.append(0)
        #     else:
        #         table.append(1)
        #
        # # 图片二值化
        # photo = Img.point(table, '1')
        bin_out=os.path.join(bin_path,video_name)
        if not os.path.exists(bin_out):
            os.mkdir(bin_out)
        # img.save(os.path.join(bin_out, "{}.jpg".format(id_name)))
        cv2.imwrite(os.path.join(bin_out, "{}.jpg".format(id_name)), img_)
        # im = Image.fromarray(img_)
        # print(im.mode)
        # # if im.mode == "F":
        # #     im = im.convert('RGB')
        # im.save(os.path.join(bin_out, "{}.jpg".format(id_name)))