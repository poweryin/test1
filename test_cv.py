import cv2
import os
import numpy as np
filename="/home/z840/data/UCF_Crimes/frame/Explosion025_x264"
# cv2.imread(filename)
frames = []
# name = "1.jpg"
frame_idx=[1,2,3,4,5]
for idx in frame_idx:
    # name = "{}{:06d}.jpg".format(rgb_prefix, idx)
    name = "{}.jpg".format(idx)
    # frames.append(cv2.imread(os.path.join(filename, name), cv2.IMREAD_COLOR))
    ima= cv2.imread(os.path.join(filename, name))
    # ima = np.transpose(ima,(2,0,1))
    # ima = ima[::,-1]
print(frames)