#!/usr/bin/python
# -*- coding:utf-8 -*-
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
import os
import re
import array
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

feature_path="/home/z840/private/yin/train_feature-/"
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
    save_fn = os.path.join(mat_path,'{}.mat'.format(video_name))
    scio.savemat(save_fn, {'idfature': li})


    li_ = np.array(li)
    x,y=li_.shape
    # if x<100:
    pca = PCA(n_components=lenth)
    li_low = pca.fit_transform(li_)

    rng = np.random.RandomState(42)
    # 构造训练样本
    n_samples = len(li_low)  # 样本总数

    outliers_fraction = 0.4  # 异常样本比例
    X_train=li_low


    # fit the model
    clf = IsolationForest(max_samples=n_samples, random_state=rng, contamination=outliers_fraction)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    scores_pred = clf.decision_function(X_train)
    print(video_name)
    print(y_pred_train)

    # LOF
    # clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
    # y_pred_train = clf.fit_predict(X_train)
    # print(video_name)
    # print(y_pred_train)
    i=i+1
print(i)



