# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import pylab as pl
import matplotlib.pyplot as plt
from itertools import cycle
import scipy.io as scio
import os
import re
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import os
import re
# similarity matrix
def cal_simi(Xn,dataLen):
    ##这个数据集的相似度矩阵，最终是二维数组
    simi = []
    for m in Xn:
        ##每个数字与所有数字的相似度列表，即矩阵中的一行
        temp = []
        for n in Xn:
            ##采用负的欧式距离计算相似度
            # s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
            s=-np.sum(np.power(m[:dataLen] - n[:dataLen], 2))
            temp.append(s)
        simi.append(temp)

    ##设置参考度，即对角线的值，一般为最小值或者中值
    p_min = np.min(simi)   ##11个中心
    p_max = np.max(simi)  ##14个中心
    p = np.median(simi)  ##5个中心
    # for i in range(dataLen):
    #     simi[i][i] = p_min
    return simi,p_min

predict_path="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_AP/mat_predict_label"
mat_path="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_AP/mat_feature"
feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_feature/"
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
    for id in feature_folder:
        filepath_ = os.path.join(filepath, id)
        f = open(filepath_, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec = np.array(array.array("f", s[20:]))
        li.append(feature_vec)
    save_fn = os.path.join(mat_path,'{}.mat'.format(video_name))
    scio.savemat(save_fn, {'idfature': li})
    txt_list_file = "/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/tufa_anno.txt"
    txt_list = open(txt_list_file).readlines()
    tmp=len(txt_list)
    for item in txt_list:
        if video_name in item:
            frame_start = int(item.split()[2])
            frame_end = int(item.split()[3])
            frame_start1 = int(item.split()[4])
            frame_end1 = int(item.split()[5])
            bg_start=int(item.split()[6])
            bg_end=int(item.split()[7])
            label = [0] * lenth
            labels_true=np.asarray(label)
            if bg_end==-1:
                labels_true[bg_start:]=1
            else:
                labels_true[bg_start:bg_end] = 1

            dataLen = len(li)
            simi_, p = cal_simi(li, dataLen)
            af = AffinityPropagation(preference=p, affinity='precomputed').fit(simi_)

            cluster_centers_indices = af.cluster_centers_indices_
            predict_label = af.labels_
            n_clusters_ = len(cluster_centers_indices)

            save_pre = os.path.join(predict_path,'{}.mat'.format(video_name))
            scio.savemat(save_pre, {'id': predict_label})
            print(video_name)
            print('Estimated number of clusters: %d' % n_clusters_)
            # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, predict_label))
            # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, predict_label))
            # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, predict_label))
            # print("Adjusted Rand Index: %0.3f"
            #       % metrics.adjusted_rand_score(labels_true, predict_label))
            # print("Adjusted Mutual Information: %0.3f"
            #       % metrics.adjusted_mutual_info_score(labels_true, predict_label))
            # print("Silhouette Coefficient: %0.3f"
            #       % metrics.silhouette_score(li, labels, metric='sqeuclidean'))
            print("labels_true")
            print(labels_true)
            print("labels")
            print(predict_label)














