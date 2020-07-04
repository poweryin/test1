# !/usr/bin/python
# coding:utf-8
import os
import math
from PIL import Image
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as scio
import mpl_toolkits.axisartist as axisartist
def sigmoid(x):
    return 1/(1+math.exp(-x))
    pass

def transform(s):
    return list(map(sigmoid, s))
    pass

video_list="/home/z840/dataset/UCF_Crimes/c3d_features/c3d_iter_test"
video_name="Abuse028_x264"
video_path = os.path.join(video_list,video_name+'_c3d.npz')
data=np.load(video_path)
idx=data['begin_idx']
scores_=data['scores']
avg_=[]
length=len(idx)

txt_list_file="/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/test.txt"
txt_list=open(txt_list_file).readlines()
for item in txt_list:
    if video_name in item:
        c=int(item.split()[2])
        d=int(item.split()[3])
        ff=int(item.split()[4])
        g=int(item.split()[5])


for i in range(length):
    avg_.extend(transform(sum(scores_[i]) / 10))

score = avg_[:]
print(score)
# plt.ion() #开启interactive mode 成功的关键函数
plt.title("028")
# fig = plt.figure(figsize=(20, 20))
# ax = axisartist.Subplot(fig, 111)
# ax.set_title(label="{}_{}".format("028"), loc='center')
# plt.plot(year,pop)
length_frame = length * 16




# fig.add_axes(ax)
# ax.axis[:].set_visible(False)
# ax.axis["x"] = ax.new_floating_axis(0, 0)
# ax.axis["x"].set_axisline_style("->", size=1.0)
# ax.axis["y"] = ax.new_floating_axis(1, 0)
# ax.axis["y"].set_axisline_style("-|>", size=1.0)
# ax.axis["x"].set_axis_direction("bottom")
# ax.axis["y"].set_axis_direction("left")
y=1
plt.xlim(0, length_frame)
plt.ylim(0, y)
plt.plot(idx, score, color='r')

f = 0
plt.text(c, f, (c), ha='center', va='bottom', fontsize=10)
plt.text(d, f, (d), ha='center', va='bottom', fontsize=10)
plt.scatter([c], [f], s=30, marker='o', color='g')
plt.scatter([d], [f], s=30, marker='o', color='g')
plt.axvline(x=c, color='g')
plt.axvline(x=d, color='g')
if ff > 0:
    plt.text(ff, f, (ff), ha='center', va='bottom', fontsize=10)
    plt.text(g, f, (g), ha='center', va='bottom', fontsize=10)
    plt.scatter([ff], [f], s=30, marker='o', color='g')
    plt.scatter([g], [f], s=30, marker='o', color='g')
    plt.axvline(x=ff, color='g')
    plt.axvline(x=g, color='g')



plt.show()
# plt.pause(0.01)

