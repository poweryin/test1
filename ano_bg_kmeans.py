# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
import os
import re
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.ensemble import IsolationForest
from scipy import stats
import scipy.io as scio
from sklearn.decomposition import PCA

def test_kmeans(data):
    x=data
    clst=cluster.KMeans(n_clusters=2)
    clst.fit(x)
    predicted_labels=clst.predict(x)
    # print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))
    # print("sum center distance %s"%clst.inertia_)
    return predicted_labels

feature_path="/home/z840/private/yin/train_bg_feature/"
mat_path="/home/z840/private/yin/train_mat/"
All_Folder = os.listdir(feature_path)
All_Folder.sort()
i=0
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
    # save_fn = os.path.join(mat_path,'{}.mat'.format(video_name))
    # scio.savemat(save_fn, {'idfature': li})


    li_ = np.array(li)
    predict_label= test_kmeans(li)
    print(video_name)
    print(predict_label)