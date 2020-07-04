# !/usr/bin/python
# coding:UTF-8

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os
save_path="/home/z840/dataset/UCF_Crimes/crop_out_30/ECO_daoxu/confidence_curve"
dataFile = '/home/z840/Downloads/ECO-master/confidencehou16.mat'
data = scio.loadmat(dataFile)
x=list(range(1299,1795))
y = list(data['confidence'])
#y=y1[::-1]
y_min=y.index(min(y))
# GT :Blue
#gt=343
#f=0
gt1=y_min
video_name="Arson016_x264_After-k"
plt.axis([1299, 1795, 0, 1])
plt.xlabel('frame count')
plt.ylabel('confidence_score')
plt.title("{}".format(video_name))
plt.plot(x, y, color='r', label=u"confidence_score")

plt.legend()
plt.draw()
plt.savefig(os.path.join(save_path, "{}.jpg".format(video_name)))
# plt.pause(0.00001)
plt.show()