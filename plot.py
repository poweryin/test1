# coding:UTF-8

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import os
save_path="/home/z840/dataset/UCF_Crimes/crop_out_30/ECO_daoxu/confidence_curve"
dataFile = '/home/z840/Downloads/ECO-master/confidence02.mat'
data = scio.loadmat(dataFile)
x=list(range(750,1241))
y1 = list(data['confidence'])
y=y1[::-1]
y_min=y.index(min(y))
gt=1125
gt1=1161
f=0
video_name="Explosion002_x264"
plt.axis([750, 1241, 0, 1])
plt.xlabel('frame count')
plt.ylabel('confidence_score')
plt.title("{}".format(video_name))
plt.plot(x, y, color='r', label=u"confidence_score")
plt.axvline(x=gt, linestyle="dotted", color='b')
plt.text(gt, f, (gt), ha='center', va='bottom', fontsize=10)
plt.scatter([gt], [f], s=30, marker='o', color='r')
plt.scatter([gt1], [f], s=30, marker='o', color='g')
plt.axvline(x=gt1, linestyle="dotted", color='g')
plt.text(gt1, f, (gt1), ha='center', va='top', fontsize=10)
plt.legend()
plt.draw()
plt.savefig(os.path.join(save_path, "{}.jpg".format(video_name)))
# plt.pause(0.00001)
plt.show()



