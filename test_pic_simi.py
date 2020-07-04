# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
import scipy.io as scio
import os
import re
import torch
feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_add_feature/"
res_feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/res_add_feature/"
All_Folder = os.listdir(feature_path)
All_Folder.sort()
# All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
for video_name in All_Folder:
    smi=[]
    filepath = os.path.join(feature_path, video_name)
    respath=os.path.join(res_feature_path,video_name)
    # read binary data
    feature_folder=os.listdir(filepath)
    lenth=len(feature_folder)
    feature_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for id in feature_folder:
        filepath_ = os.path.join(filepath, id)
        respath_=os.path.join(respath,id)
        # bg clip feature
        f = open(filepath_, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec_bg = np.array(array.array("f", s[20:]))
        feature_vec=feature_vec_bg.reshape(1,4096)
        # res clip feature
        ff = open(respath_, "rb")
        ss = ff.read()
        ff.close()
        (n1, c1, l1, h1, w1) = array.array("i", ss[:20])
        feature_vec_res = np.array(array.array("f", ss[20:]))
        feature_vec_=feature_vec_res.reshape(4096,1)
        corr_simi = np.dot(feature_vec,feature_vec_)
        similar=int(corr_simi.reshape(1))
        print(similar)
        smi.append(similar)
    id=smi
    id_=np.array(id)
    # 1----10 zhengxu
    index_=np.argsort(-id_)
    min_smi=min(id)
    min_index=id.index(max(id))
    print(min_smi)
    print(min_index)
    print(index_)
        # bg.append(feature_vec)
    # save_fn = os.path.join(mat_path,'{}.mat'.format(video_name))
    # scio.savemat(save_fn, {'idfature': li})