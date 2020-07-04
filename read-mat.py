# coding:UTF-8


import scipy.io as scio

dataFile = '/home/z840/disktend/oldhome/poweryin/AnomalyDetection_MIL/Paper_Results/6_C3D_Final_L1L2.mat'
data = scio.loadmat(dataFile)
idx=data['confidence']
print(data)