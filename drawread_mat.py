# coding:utf-8
import scipy.io as scio
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import os
mat_path='/home/z840/dataset/UCF_Crimes/crop_out_30/mat_new/Arson016_x264.mat'
save_path="/home/z840/dataset/UCF_Crimes/crop_out_30/simi_curve_new/"
video_name=mat_path.split('/')[-1].split('.')[0]
frame_tmp = scio.loadmat(mat_path)
t=frame_tmp['id_patch']
x,y=frame_tmp['id_patch'].shape
# gt=142
# gt=94
# gt=65
gt=200
# gt1=200
# gt1=219
# gt=255
f=0
for i in range(16):
    i_patch=frame_tmp['id_patch'][i,:]
    Total_frames=y
    x = np.linspace(1, Total_frames, Total_frames)
    plt.axis([0, Total_frames+1, 0, 100])
    plt.xlabel('frame count')
    plt.ylabel('simi_score')
    plt.title("{}_{}.jpg".format(video_name,i))
    plt.plot(x, i_patch, color='r', label=u"simi_score")
    plt.text(gt, f, (gt), ha='center', va='bottom', fontsize=10)
    # plt.text(gt1, f, (gt1), ha='center', va='bottom', fontsize=10)
    plt.scatter([gt], [f], s=30, marker='o', color='b')
    # plt.scatter([gt1], [f], s=30, marker='o', color='g')
    plt.axvline(x=gt, linestyle="dotted",color='b')
    # plt.axvline(x=gt1, linestyle="dotted", color='b')
    # plt.axvline(x=gt1, linestyle="dotted", color='g')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_path, "{}_{}.jpg".format(video_name,i)))
    plt.pause(0.00001)
    plt.show()

# arr= np.linspace(1,5,20)
# p_=8.0
# p_arr = np.concatenate((arr,[p_]))
# frame_count = list(chain.from_iterable((p_arr))
# print(frame_count)

