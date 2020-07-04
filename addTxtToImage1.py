#给图片上添加文字注释

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
import shutil
import numpy as np
import re

def addTextTOImage(imageSrc, newImage):
    #
    image = Image.open(imageSrc)
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 20)
    #draw.text((5, 5), "Anomaly!!!", (255), font=font)
    draw.text((5, 5), "Anomaly!!!", (255,255,0))
    ImageDraw.Draw(image)
    # 保存图片
    image.save(newImage)


def addTextTOImage1(imageSrc, newImage):
    #
    image = Image.open(imageSrc)
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("C:\\WINDOWS\\Fonts\\SIMYOU.TTF", 20)
    # draw.text((5, 5), "Anomaly!!!", (255), font=font)
    draw.text((10, 10), "GT-Anomaly!!!", (75,200,86))
    ImageDraw.Draw(image)
    # 保存图片
    image.save(newImage)


def matLoad():
    # marginsPed2 = [range(60,180),range(94,180),range(0,146),range(30,180),range(0,129),range(0,159),range(45,180),range(0,180),range(0,120),range(0,150),range(0,180),range(87,180)]
    marginsPed1 = [range(59, 152), range(49, 175), range(90, 200), range(30, 168), [range(4, 90), range(139, 200)],
                   [range(0, 100), range(109, 200)], range(0, 175),
                   range(0, 94), range(0, 48), range(0, 140), range(69, 165), range(129, 200), range(0, 156),
                   range(0, 200), range(137, 200), range(122, 200),
                   range(0, 47), range(53, 120), range(63, 138), range(44, 175), range(30, 200), range(15, 107),
                   range(7, 165), range(49, 171), range(39, 135),
                   range(15, 107), range(7, 165), range(49, 171), range(39, 135), range(76, 144), range(10, 122),
                   range(105, 200), [range(0, 15), range(44, 113)],
                   range(174, 200), range(0, 180), [range(0, 52), range(64, 115)], range(4, 115), range(0, 121),
                   range(85, 200), range(14, 108)]
    videoPath = "E:/Dataset/Anomaly/UCSD/UCSDped1/test_/"
    video = os.listdir("E:/Dataset/Anomaly/UCSD/UCSDped1/test_/")
    newPath = "E:/Dataset/Anomaly/UCSD/UCSDped1/test1/"
    count = 0
    index = 0
    for images in video:
        if index >= len(marginsPed1):
            break
        for image in os.listdir(os.path.join(videoPath, images)):
            # 跳过隐藏文件
            if image == "._.DS_Store" or image == ".DS_Store":
                continue
            else:
                image_ = videoPath + images + '/' + image
                imageNew = newPath + images
                if not os.path.exists(imageNew):
                    os.mkdir(imageNew)
                if len(marginsPed1[index]) == 2:
                    anomarlyIndex = list(marginsPed1[index][0])
                    temp = list(marginsPed1[index][1])
                    anomarlyIndex.extend(temp)
                else:
                    anomarlyIndex = list(marginsPed1[index])
                # if count in marginsPed2[index]:
                if count in anomarlyIndex:
                    addTextTOImage(image_, os.path.join(imageNew, image))
                else:
                    shutil.copyfile(image_, os.path.join(imageNew, image))
                count += 1
        index += 1
        count = 0

def npyLoad():

    labelPath = "/data/UCF_Crimes/c3d_features/curve_tmp/ind_test.txt"
    videoPath = "/data/UCF_Crimes/test_frames/"
    donePath = "/data/UCF_Crimes/test_addtxt/"
    donePath_="/data/UCF_Crimes/test_add_final/"
    #frameLabeled = os.listdir(labelPath)
    videos = os.listdir(videoPath)
    txt_list = open(labelPath).readlines()
    # f = open(labelPath, "r")
    # frameLabeled = f.readlines()
    for item in txt_list:
        video =item.split(".",1)[0]
        frame_path = os.path.join(videoPath, video)
        pictures=os.listdir(frame_path)
        total_frames = len(os.listdir(frame_path))
        _,ind_label, gt = item.split(":", maxsplit=2)
        ind_label=ind_label[:-3]
        gt=gt[:-1]
        ind_label = list(map(int, ind_label.split()))
        gt = list(map(int, gt.split()))



        # for labels, images in zip(tmp, video):
        #     # label = np.matrix.tolist(np.load(os.path.join(labelPath, labels)))

        image = os.listdir(os.path.join(videoPath, video))
        image.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        pathNew = os.path.join(donePath,  video)

        if not os.path.exists(pathNew):
            os.mkdir(pathNew)

        for x,y in zip(tmp, image):
        # for x,y in enumerate(image):
            image_ = videoPath + video + "/" + y
            if x == 0:
                shutil.copyfile(image_, os.path.join(pathNew, y))
            else:
                addTextTOImage(image_, os.path.join(pathNew, y))




if __name__ == "__main__":
    #matLoad()
    npyLoad()









