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
import os
import re
feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/add/bg_feature/test"
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
    # t=0
    for id in feature_folder:
        filepath_ = os.path.join(filepath, id)
        f = open(filepath_, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec = np.array(array.array("f", s[20:]))
        li.append(feature_vec)

# label=[0]*lenth
# bg_start=73
# labels_true = np.asarray(label)
# labels_true[bg_start:] = 1

label=[0]*lenth
bg_start=109
labels_true = np.asarray(label)
labels_true[bg_start:] = 1


# def cal_simi(Xn,dataLen):
#     ##这个数据集的相似度矩阵，最终是二维数组
#     simi = []
#     for m in Xn:
#         ##每个数字与所有数字的相似度列表，即矩阵中的一行
#         temp = []
#         for n in Xn:
#             ##采用负的欧式距离计算相似度
#             # s =-np.sqrt((m[0]-n[0])**2 + (m[1]-n[1])**2)
#             s=-np.sqrt(np.sum(np.power(m[:dataLen] - n[:dataLen], 2)))
#             temp.append(s)
#         simi.append(temp)
#
#     ##设置参考度，即对角线的值，一般为最小值或者中值
#     p_min = np.min(simi)   ##11个中心
#     p_max = np.max(simi)  ##14个中心
#     p = np.median(simi)  ##5个中心
#     for i in range(dataLen):
#         simi[i][i] = p_min
#     return simi,p_min

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




dataLen = len(li)
# simi_,p=cal_simi(li,dataLen)
simi_,p=cal_simi(li,dataLen)

# Compute Affinity Propagation
# af = AffinityPropagation(preference=-50).fit(li)
af = AffinityPropagation(preference=p*2.9,affinity='precomputed').fit(simi_)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(li, labels, metric='sqeuclidean'))
print("labels_true")
print(labels_true)
print("labels")
print(labels)
# Plot result
# pl.close('all')
# plt.figure(1)
# plt.clf()
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)