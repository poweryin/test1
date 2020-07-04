import cv2
import numpy as np
import math
import os
import re
import scipy.io as scio
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
feature_path="/home/z840/dataset/UCF_Crimes/shang/"
save_path="/home/z840/dataset/UCF_Crimes/shang_save"
All_Folder = os.listdir(feature_path)
All_Folder.sort()
# All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
for video_name in All_Folder:
    li = []
    filepath = os.path.join(feature_path, video_name)
    # read binary data
    feature_folder=os.listdir(filepath)
    lenth=len(feature_folder)
    feature_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    shang=[]
    frame_len=len(feature_folder)
    for id in feature_folder:
        filepath_ = os.path.join(filepath, id)


        tmp = []
        for i in range(256):
            tmp.append(0)
        val = 0
        k = 0
        res = 0
        image = cv2.imread(filepath_,0)
        img = np.array(image)
        # img=img[120:239,160:319]
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i][j]
                tmp[val] = float(tmp[val] + 1)
                k =  float(k + 1)
        for i in range(len(tmp)):
            tmp[i] = float(tmp[i] / k)
        for i in range(len(tmp)):
            if(tmp[i] == 0):
                res = res
            else:
                res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
        # print(res)
        shang.append(res)

    gt=526
    f=0
    print(shang)
    x = np.linspace(1, frame_len, frame_len)
    plt.axis([0, frame_len + 1, 0, 10])
    plt.xlabel('frame count')
    plt.ylabel('shang_score')
    plt.title("{}".format(video_name))
    plt.plot(x, shang, color='r', label=u"shang_score")
    plt.axvline(x=gt, linestyle="dotted", color='b')
    plt.text(gt, f, (gt), ha='center', va='bottom', fontsize=10)
    plt.scatter([gt], [f], s=30, marker='o', color='b')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_path, "{}.jpg".format(video_name)))
    # plt.pause(0.00001)
    plt.show()

