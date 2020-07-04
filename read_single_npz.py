# !/usr/bin/python
# -*- coding:utf8 -*-
#
# import caffe
# caffe.set_device(0)
# caffe.set_mode_gpu()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

path="/data/UCF_Crimes/c3d_features/c3d_iter_1000/Fighting003_x264_c3d.npz"
data=np.load(path)
idx=data["begin_idx"]
scores=data["scores"]
score = []
length=len(idx)
print(len(idx))
# print(scores)


for i in range(length):
    # print(idx)
    # print(scores[i])
    # print(scores[i].shape[0])
    score.append(abs(np.sum(scores[i])/10))
y = max(score) + 0.2
print(idx)
print(score)


# year=[1950,1970,1990,2010]
# pop=[2.518,3.68,5.23,6.97]
#2.散点图,只是用用scat函数来调用即可
plt.title('curve figure')
# plt.plot(year,pop)
plt.plot(idx,score,color='r')
# plt.plot(idx, score,  color='r',markerfacecolor='blue',marker='o')
# for a, b in zip(idx, score):
#     plt.text(a, b, (a),ha='center', va='bottom', fontsize=10)

# plt.text(c, f, (c), ha='center', va='bottom', fontsize=10)
# plt.text(d, f, (d), ha='center', va='bottom',fontsize=10)
# plt.scatter([c], [f], s=30,marker='o',color='g')
# plt.scatter([d], [f], s=30,marker='o',color='g')
# plt.axvline(x=c,color='g')
# plt.axvline(x=d,color='g')
length_frame=length*16
plt.axis([0, length_frame, 0, y])
plt.legend()
plt.xlabel('frame number')
plt.ylabel('score')
# plt.savefig('{}.jpg'.format(Explosion025_x264)
plt.savefig('Fighting003_x264.jpg')
plt.show()

