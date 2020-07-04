import os
import cv2
import numpy as np
def image_label(video_filename):
    labels=[]
    length=len(os.listdir(video_filename))
    for i in range(0,length):
        if i in range(1820,length):
            label = 1
            labels.append(label)
        else:
            label=0
            labels.append(label)
    np.save(video_filename+".npy", labels)
    print(labels)

if __name__ == '__main__':
    path="/home/z840/poweryin/ano_pred_cvpr2018-master/Data/fight/Fighting003_x264"

    image_label(path)