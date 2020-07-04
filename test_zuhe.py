import argparse
import glob
import re
import cv2
import numpy as np
import os


def show_in_one(images, show_size=(336, 336), blank_size=4, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    return merge_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate mouse interaction with images')
    savepath_="/home/z840/dataset/UCF_Crimes/crop_test_out_30/save_zuhe"
    parser.add_argument("-i", "--input", help="Input directory.")
    args = parser.parse_args()
    path = args.input
    if path is None:
        test_dir = "/home/z840/dataset/UCF_Crimes/crop_test_out_30/to9"
        test_path=os.listdir(test_dir)
        test_path.sort(key=lambda x: list(map(int, x.split('.')[0].split('_'))))
        # test_path.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        # path = test_dir

    debug_images = []

    for infile in test_path:
        infile_path=os.path.join(test_dir,infile)
        # ext = os.path.splitext(infile)[1][1:]  # get the filename extenstion
        # if ext == "png" or ext == "jpg" or ext == "bmp" or ext == "tiff" or ext == "pbm":
        #     print(infile)
        img = cv2.imread(infile_path)
        if img is None:
            continue
        else:
            debug_images.append(img)

    merge_im=show_in_one(debug_images)

    id_=8
    cv2.imwrite(os.path.join(savepath_, "{}.jpg".format(id_)), merge_im)
    cv2.waitKey(0)
    cv2.destroyWindow()


# import PIL.Image as Image
# import os
#
# IMAGES_PATH = '/home/z840/dataset/UCF_Crimes/crop_test_out_30/to9img/'  # 图片集地址
# IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
# IMAGE_SIZE = 80  # 每张小图片的大小
# IMAGE_ROW = 3  # 图片间隔，也就是合并成一张图后，一共有几行
# IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
# IMAGE_SAVE_PATH = '/home/z840/dataset/UCF_Crimes/crop_test_out_30/zuhe/final.jpg'  # 图片转换后的地址
#
# # 获取图片集地址下的所有图片名称
# image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#                os.path.splitext(name)[1] == item]
# image_names.sort()
# # 简单的对于参数的设定和实际图片集的大小进行数量判断
# if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
#     raise ValueError("合成图片的参数和要求的数量不能匹配！")
#
#
# # 定义图像拼接函数
# def image_compose():
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
#             to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
#     return to_image.save(IMAGE_SAVE_PATH)  # 保存新图
#
#
# image_compose()  # 调用函数
