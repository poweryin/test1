# coding:utf-8
import cv2 as cv
import numpy as np
import os
import re
import scipy.io as scio
import matplotlib.pyplot as plt
# src = cv.imread('/home/z840/dataset/UCF_Crimes/test/RoadAccidents002_x264/000001.jpg', cv.IMREAD_GRAYSCALE)

def pic_sub(dest):
    white_num=0
    height, width = dest.shape
    for i in range(height):
        for j in range(width):
            if dest[i, j] == 255:
                white_num += 1
    return white_num









test_video_path="/home/z840/dataset/UCF_Crimes/test/"
result_path="/home/z840/dataset/UCF_Crimes/test_xiangsu/xiangsu_mat/"
curve_path="/home/z840/dataset/UCF_Crimes/test_xiangsu/xiangsu_curve/"
video_=os.listdir(test_video_path)
video_.sort()
for video_name in video_:
    frame_dir=os.path.join(test_video_path,video_name)
    result_=os.path.join(result_path,video_name)
    # if not os.path.exists(result_):
    #     os.makedirs(result_)

    video_frame = os.listdir(frame_dir)
    video_frame.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    mean_ = []
    std_ = []
    white_=[]
    for im_name in video_frame:
        frame_path=os.path.join(frame_dir,im_name)
        src = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
        # white_xiangsu=pic_sub(src)
        cv.imshow('input', src)
        # 最大最小值和相应的位置
        min, max, minLoc, maxLoc = cv.minMaxLoc(src)
        # 均值和标准差
        mean, std = cv.meanStdDev(src)
        t=mean[0][0]

        mean_.append(mean[0][0])
        std_.append(std[0][0])
        # white_.append(white_xiangsu)

    # dataNew = os.path.join(result_path,"{}".format(video_name))
    # scio.savemat(dataNew, {'mean': mean_,'std':std_})
    Total_frames=len(mean_)
    x = np.linspace(1, Total_frames, Total_frames)
    plt.axis([0, Total_frames+1, 0, 400])
    plt.xlabel('frame count')
    plt.ylabel('score')
    # introduce GT
    # txt_list_file = "/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/test.txt"
    # txt_list = open(txt_list_file).readlines()
    # for item in txt_list:
    #     if video_name in item:
    #         c = int(item.split()[2])
    #         d = int(item.split()[3])
    #         ff = int(item.split()[4])
    #         g = int(item.split()[5])

    # f = 0
    # plt.text(c, f, (c), ha='center', va='bottom', fontsize=10)
    # plt.text(d, f, (d), ha='center', va='bottom', fontsize=10)
    # plt.scatter([c], [f], s=30, marker='o', color='b')
    # plt.scatter([d], [f], s=30, marker='o', color='b')
    # plt.axvline(x=c, color='b')
    # plt.axvline(x=d, color='b')
    # if ff > 0:
    #     plt.text(ff, f, (ff), ha='center', va='bottom', fontsize=10)
    #     plt.text(g, f, (g), ha='center', va='bottom', fontsize=10)
    #     plt.scatter([ff], [f], s=30, marker='o', color='b')
    #     plt.scatter([g], [f], s=30, marker='o', color='b')
    #     plt.axvline(x=ff, color='b', ls='--')
    #     plt.axvline(x=g, color='b', ls='--')
    #     plt.legend()




    # max_mean = max(mean_)
    # max_std = max(std_)

    plt.plot(x, mean_, color='r',label=u"mean" )
    plt.legend()
    plt.draw()
    # plt.figure(2)
    plt.plot(x,std_, color='g',label=u"std")
    plt.legend()
    plt.title("{}".format(video_name))

    plt.plot(x,white_, color='b',label=u"white_num")
    plt.legend()

    plt.savefig(os.path.join(curve_path,"{}".format(video_name)))
    plt.pause(0.00001)
    plt.show()

    print(mean_)
    print(std_)


