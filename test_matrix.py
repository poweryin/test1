# coding:utf-8
import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import array
t=torch.zeros([5,4096])
print(t.shape)
feature_path="/home/z840/dataset/UCF_Crimes/caffe_train1/"
All_Folder = os.listdir(feature_path)
All_Folder.sort()
for i in All_Folder:
    li = []
    filepath = os.path.join(feature_path, i)
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

arr=np.array(li)
print(arr)