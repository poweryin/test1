# !/usr/bin/python
# -*- coding:utf8 -*-
import os
import math
from PIL import Image
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio

# video_list='/home/z840/poweryin/GCN-Anomaly-Detection-master/c3d_features/hecheng/'
video_list='/data/UCF_Crimes/c3d_features/c3d_iter_1000/'
out_path='/data/UCF_Crimes/c3d_features/curve_tmp/'
txt_list_file="/data/UCF_Crimes/Anomaly_Detection_splits/test.txt"
txt_list=open(txt_list_file).readlines()
# txt_list = [item.strip() for item in txt_list]

def sigmoid(x):
    return 1/(1+math.exp(-x))
    pass

def transform(s):
    return list(map(sigmoid, s))
    pass
#
for item in txt_list:
    c=int(item.split()[2])
    d=int(item.split()[3])
    video_name=item.split(".",1)[0]
dirs = os.listdir(video_list)
for video in dirs:
# video_name="Explosion050_x264"
    video_path = os.path.join(video_list,video_name+'_c3d.npz')
    data=np.load(video_path)
    idx=data['begin_idx']
    scores=data['scores']
    score = []
    x_idx=[]
    y_score=[0]*idx[-1]
    length=len(idx)
    flag = ["average", "median"]
    max_, min_, avg_, median_ = [], [], [], []
        # print(len(idx))

    print(scores)
    # for i in range(0,length):

    for i in range(length):
            # score.append(abs(np.sum(scores[i])/10))
            # max_.append(max(transform(scores[i])))
            # min_.append(min(transform(scores[i])))
        avg_.extend(transform(sum(scores[i])/10))
        tmp = sorted(scores[i])
        median_.append(transform((tmp[4]+tmp[5])/2))
        # score.extend([avg_, median_])
    score=avg_[:]
    save_fn = '{}.mat'.format(video_name)



    # year=[1950,1970,1990,2010]
    # pop=[2.518,3.68,5.23,6.97]
    #2.散点图,只是用scat函数来调用即可
    for i in range(1):
        plt.title("{}_{}".format(video_name.split('.',1)[0],flag[i]))
            # plt.plot(year,pop)
        length_frame = length * 16
            # y = max(score[i]) +0.02
            # plt.axis([0, length_frame,0,y])
        plt.plot(idx,score,color='r')
        # f = 0
        # plt.text(c, f, (c), ha='center', va='bottom', fontsize=10)
        # plt.text(d, f, (d), ha='center', va='bottom', fontsize=10)
            # plt.scatter([c], [f], s=30, marker='o', color='g')
            # plt.scatter([d], [f], s=30, marker='o', color='g')
        # plt.axvline(x=92, color='g')
        # plt.axvline(x=276, color='g')
        # plt.axvline(x=594, color='b')
        # plt.axvline(x=998, color='b')

        plt.legend()
        plt.xlabel('frame number')
        plt.ylabel('score')
        plt.savefig('{}_{}.jpg'.format(video_name, flag[i]))
            # plt.show()
        plt.pause(10)  # 显示秒数
        plt.close()
