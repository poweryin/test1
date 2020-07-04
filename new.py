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
out_path='/data/UCF_Crimes/c3d_features/curve_t/'
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
    tmp_gt = ""
    for s in a:
        tmp_gt += s
        tmp_gt +=" "

    string = "{}:{} {}:{}\n".format(video, tmpstr, "gt", tmp_gt)
    return string

with open(save_path, "w") as f:
    res = []
    threhold = 0.57
    for item in txt_list:
        print(item.split(maxsplit=2))
        arr= item.split(' ', maxsplit=2)
        video, _, linespace = arr[0],arr[1],arr[2]
        video_name = item.split(".", 1)[0]
        video_path = os.path.join(video_list, video_name + '_c3d.npz')
        data = np.load(video_path)
        idx = data['begin_idx']
        scores = data['scores']
        x_idx = []
        y_score = [0] * idx[-1]
        length = len(idx)
        flag = ["average", "median"]
        max_, min_, avg_, median_ = [], [], [], []
        for i in range(length):
            # score.append(abs(np.sum(scores[i])/10))
            # max_.append(max(transform(scores[i])))
            # min_.append(min(transform(scores[i])))
            avg_.extend(transform(sum(scores[i])/10))
            tmp = sorted(scores[i])
            median_.append(transform((tmp[4]+tmp[5])/2))
        score = avg_[:]
        res.append(save_idx(idx, score, threhold, video, linespace))
    f.writelines(res)