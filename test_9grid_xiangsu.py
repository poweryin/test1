# coding:utf-8
from PIL import Image
import os
import re
import cv2 as cv
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
def fill_image(image):
    width, height = image.size

    new_len = max(width, height)

    # 将新图片是正方形，长度为原宽高中最长的
    new_image = Image.new(image.mode, (new_len, new_len), color='white')

    # 根据两种不同的情况，将原图片放入新建的空白图片中部
    if width > height:
        new_image.paste(image, (0, int((new_len - height) / 2)))
    else:
        new_image.paste(image, (int((new_len - width) / 2), 0))
    return new_image


def cut_image(image):
    width, height = image.size
    # item_width = int(width / 3)
    item_width = int(width / 4)
    # 保存每一个小切图的区域
    box_list = []

    for i in range(4):
        for j in range(4):
            # 切图区域是矩形，位置由对角线的两个点(左上和右下)确定
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(image_list, out_dir,pic_id):
    for (index, image) in enumerate(image_list):
        image.save(f"{out_dir}/{pic_id}_{index+1}.jpg", "JPEG")


def to9img(file_path, video_name):
    pic_id = file_path.split('/')[-1].split('.')[0]
    image = Image.open(file_path)
    image = image.resize((240, 240))
    image = fill_image(image)
    image_list = cut_image(image)
    out_dir_="/home/z840/dataset/UCF_Crimes/crop_test_out_30/to9img"
    out_dir=os.path.join(out_dir_,video_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_images(image_list, out_dir,pic_id)


if __name__ == '__main__':
    # to9img('/home/z840/dataset/UCF_Crimes/crop_test_out_30/bg/Arson016_x264/1451.jpg')
    # pic_path="/home/z840/dataset/UCF_Crimes/crop_out_30/res_test/"
    pic_path="/home/z840/dataset/UCF_Crimes/crop_test_out_30/to9img/"
    curve_path="/home/z840/dataset/UCF_Crimes/crop_test_out_30/curve/"
    All_Folder = os.listdir(pic_path)
    All_Folder.sort()
    # All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for video_name in All_Folder:
        filepath = os.path.join(pic_path, video_name)
        # filepath_ = os.path.join(res_path, video_name)
        # diffpath_ = os.path.join(result_path, video_name)
        # if not os.path.exists(diffpath_):
        #     os.makedirs(diffpath_)
        # read binary data
        pic_folder = os.listdir(filepath)
        lenth = len(pic_folder)
        # pic_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        pic_folder.sort(key=lambda x: list(map(int, x.split('.')[0].split('_'))))
        mean_ = []
        std_ = []
        for id in pic_folder:
            id_ = int(id.split('.')[0].split('_')[-1])
            if id_!=6:
                continue
            idpath_ = os.path.join(filepath, id)
            # to9img(idpath_,video_name)
            src = cv.imread(idpath_, cv.IMREAD_GRAYSCALE)
            cv.imshow('input', src)
            # 最大最小值和相应的位置
            min, max, minLoc, maxLoc = cv.minMaxLoc(src)
            # 均值和标准差
            mean, std = cv.meanStdDev(src)
            mean_.append(mean[0][0])
            std_.append(std[0][0])
        Total_frames = len(mean_)
        x = np.linspace(1, Total_frames, Total_frames)
        plt.axis([0, Total_frames + 1, 0, 400])
        plt.xlabel('frame count')
        plt.ylabel('score')
        plt.plot(x, mean_, color='r', label=u"mean")
        plt.legend()
        plt.draw()
        plt.savefig(os.path.join(curve_path, "5_{}".format(video_name)))
        plt.pause(0.00001)
        plt.show()