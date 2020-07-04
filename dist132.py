# !/usr/bin/python
# coding:UTF-8
# 从589帧往前回溯,绘制异常目标框每帧移动的距离
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os
save_path="/home/z840/dataset/UCF_Crimes/test_demo/anomal_track/distance_curve"
dataFile = '/home/z840/Downloads/ECO-master/dis132.mat'
data = scio.loadmat(dataFile)
x=list(range(1,490))
y = list(data['dist'])
#y=y1[::-1]
y_min=y.index(min(y))
# GT :Blue
gt=369
# gt1=370
f=0
#gt1=y_min
video_name="RoadAccidents132_x264"
plt.axis([1, 490, 0, 3.5])
plt.xlabel('frame count')
plt.ylabel('distance')
plt.title("{}".format(video_name))
plt.plot(x, y, color='r', label=u"distance")
plt.axvline(x=gt, linestyle="dotted", color='b')
plt.text(gt, f, (gt), ha='center', va='bottom', fontsize=10)
plt.scatter([gt], [f], s=30, marker='o', color='r')
# plt.scatter([gt1], [f], s=30, marker='o', color='g')
# plt.axvline(x=gt1, linestyle="dotted", color='g')
# plt.text(gt1, f, (gt1), ha='center', va='top', fontsize=10)
plt.legend()
plt.draw()
plt.savefig(os.path.join(save_path, "{}.jpg".format(video_name)))
# plt.pause(0.00001)
plt.show()