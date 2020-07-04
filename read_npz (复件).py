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
import mpl_toolkits.axisartist as axisartist
video_list='/data/UCF_Crimes/c3d_features/c3d_iter_test/'
# video_list='/home/z840/poweryin/GCN-Anomaly-Detection-master/c3d_features/c3d_iter_1000/'
out_path='/data/UCF_Crimes/c3d_features/curve_tt/'
txt_list_file="/data/UCF_Crimes/Anomaly_Detection_splits/test.txt"
txt_list=open(txt_list_file).readlines()
# txt_list = [item.strip() for item in txt_list]
save_path = "/data/UCF_Crimes/c3d_features/curve_tmp/ind.txt"

def sigmoid(x):
    return 1/(1+math.exp(-x))
    pass

def transform(s):
    return list(map(sigmoid, s))
    pass

def save_idx(idx, score, threhold, video, linespace):
    tmp = []
    flag = True
    for ind, s in zip(idx, score):
        if s >= threhold and flag:
            tmp.append(ind)
            flag = False
        if s <= threhold and not flag:
            tmp.append(ind)
            flag = True
    tmpstr = ""
    for s in tmp:
        tmpstr += (str(s)+" ")

    a = linespace.strip().split()
    if a[-1] == '-1':
        a.pop(-1)
        a.pop(-1)
    a.pop(0)
    tmp_gt = ""
    for s in a:
        tmp_gt += s
        tmp_gt +=" "

    string = "{}:{} {}:{}\n".format(video, tmpstr, "gt", tmp_gt)
    return string
#
# with open(save_path, "w") as f:
#     res = []
#     threhold = 0.57
#     for item in txt_list:
#         arr = item.split(' ', maxsplit=2)
#         video, _, linespace = arr[0], arr[1], arr[2]
#         video_name = item.split(".", 1)[0]
#         video_path = os.path.join(video_list, video_name + '_c3d.npz')
#         data = np.load(video_path)
#         idx = data['begin_idx']
#         scores = data['scores']
#         x_idx = []
#         y_score = [0] * idx[-1]
#         length = len(idx)
#         flag = ["average", "median"]
#         max_, min_, avg_, median_ = [], [], [], []
#         for i in range(length):
#             # score.append(abs(np.sum(scores[i])/10))
#             # max_.append(max(transform(scores[i])))
#             # min_.append(min(transform(scores[i])))
#             avg_.extend(transform(sum(scores[i])/10))
#             tmp = sorted(scores[i])
#             median_.append(transform((tmp[4]+tmp[5])/2))
#         score = avg_[:]
#         res.append(save_idx(idx, score, threhold, video, linespace))
#     f.writelines(res)

for item in txt_list:
    c=int(item.split()[2])
    d=int(item.split()[3])
    ff=int(item.split()[4])
    g=int(item.split()[5])
    video_name=item.split(".",1)[0]
    video, _, linespace = item.split(maxsplit=2)
# dirs = os.listdir(video_list)
# for video_name in dirs:
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
    threhold = 0.57
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
    y = max(score) + 0.02
    save_fn = '{}.mat'.format(video_name)
    # save_array = np.array([1, 2, 3, 4])
    # scio.savemat(save_fn, {'array': save_array})  # 和上面的一样，存在了array变量的第一行

    save_array_x = idx
    save_array_y = score
    scio.savemat(save_fn, {'idx': save_array_x, 'score': save_array_y})

    #2.散点图,只是用scat函数来调用即可
    for i in range(1):
        plt.title("{}_{}".format(video_name.split('.',1)[0],flag[i]))
        fig = plt.figure(figsize=(20, 20))
        ax = axisartist.Subplot(fig, 111)
        ax.set_title(label="{}_{}".format(video_name.split('.',1)[0],flag[i]), loc='center')
        # plt.plot(year,pop)
        length_frame = length * 16

        # plt.axis([0, length_frame,0,y])


        # 使用axisartist.Subplot方法创建一个绘图区对象ax

        # 将绘图区对象添加到画布中
        fig.add_axes(ax)
        ax.axis[:].set_visible(False)
        ax.axis["x"] = ax.new_floating_axis(0, 0)
        ax.axis["x"].set_axisline_style("->", size=1.0)
        ax.axis["y"] = ax.new_floating_axis(1, 0)
        ax.axis["y"].set_axisline_style("-|>", size=1.0)
        ax.axis["x"].set_axis_direction("bottom")
        ax.axis["y"].set_axis_direction("left")
        plt.xlim(0, length_frame)
        plt.ylim(0, y)
        ax.plot(idx, score, color='r')
        # ax.set_xlabel('frame number')  # 为子图设置横轴标题
        # ax.set_ylabel('score')  # 为子图设置纵轴标题
        # plt.xlabel('frame number')
        # plt.ylabel("aaaaa")
        # ax.text(x=-2, y=0.5, s="score", fontdict={'size': 15, 'color': 'black', 'rotation': 90})
        # ax.text(x=0.8, y=-0.1, s="frame number", fontdict={'size': 15, 'color': 'black', 'rotation': 0})
        #
        f = 0
        plt.text(c, f, (c), ha='center', va='bottom', fontsize=10)
        plt.text(d, f, (d), ha='center', va='bottom', fontsize=10)
        plt.scatter([c], [f], s=30, marker='o', color='g')
        plt.scatter([d], [f], s=30, marker='o', color='g')
        plt.axvline(x=c, color='g')
        plt.axvline(x=d, color='g')
        if ff>0:
            plt.text(ff, f, (ff), ha='center', va='bottom', fontsize=10)
            plt.text(g, f, (g), ha='center', va='bottom', fontsize=10)
            plt.scatter([ff], [f], s=30, marker='o', color='g')
            plt.scatter([g], [f], s=30, marker='o', color='g')
            plt.axvline(x=ff, color='g')
            plt.axvline(x=g, color='g')


        lim1 = [threhold]*20000
        plt.plot(lim1, "b--")
        plt.text(f, threhold, (threhold), ha='center', va='bottom', fontsize=10)




        # plt.show()
        # plt.pause(0.001)  # 显示秒数
        plt.savefig('/data/UCF_Crimes/c3d_features/curve_tt/{}_{}.jpg'.format(video_name, flag[i]))
        plt.close()

