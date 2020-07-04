# coding:utf-8
from PIL import Image
import os


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


def save_images(image_list, out_dir):
    for (index, image) in enumerate(image_list):
        image.save(f"{out_dir}/{index}.jpg", "JPEG")


def to9img(file_path, out_dir='.'):
    image = Image.open(file_path)
    image = image.resize((320, 320))
    image = fill_image(image)
    image_list = cut_image(image)
    out_dir="/home/z840/dataset/UCF_Crimes/crop_test_out_30/to9"
    save_images(image_list, out_dir)


if __name__ == '__main__':
    # to9img('/home/z840/dataset/UCF_Crimes/crop_out_30/bg/Explosion016_x264/000455.jpg')
    # to9img('/home/z840/dataset/UCF_Crimes/crop_out_30/res_add/RoadAccidents132_x264/560.jpg')
    to9img("/home/z840/dataset/UCF_Crimes/crop_out_30/simi/Arson016_x264/1025.jpg")