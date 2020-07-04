# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
import os
import re
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN

def test_kmeans(*data):
    x,labels_true=data
    clst=cluster.KMeans(n_clusters=2)
    clst.fit(x)
    predicted_labels=clst.predict(x)
    ARI = adjusted_rand_score(labels_true, predicted_labels)
    # print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))
    # print("sum center distance %s"%clst.inertia_)
    return predicted_labels,ARI

def test_DBSCAN(*data):
    x, labels_true = data
    db = DBSCAN(eps=27, min_samples=15, metric='euclidean')  # 欧式距离，样本点数量5，半径0.2
    y_db = db.fit_predict(x)
    return y_db
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
        # t+=1;
        # if t==86:
        #     break

# centers=[[1,1],[2,2],[1,2],[10,20]]
# x,labels_true=create_data(centers,1000,0.5)
# x=li

label=[0]*lenth
bg_start=117
labels_true = np.asarray(label)
labels_true[bg_start:137] = 1


# predict,ari=test_kmeans(li,labels_true)
predict=test_DBSCAN(li,labels_true)
print("true label:")
print(labels_true)
print(len(labels_true))
print("predict_label")
print(predict)
print(len(predict))

