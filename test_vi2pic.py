# coding:utf-8
import cv2
import os
import re
img_root = '/home/z840/dataset/UCF_Crimes/test/RoadAccidents002_x264/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 30    #保存视频的FPS，可以适当调整
size=(320,240)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('/home/z840/dataset/UCF_Crimes/test_output/3.avi',fourcc,fps,size)#最后一个是保存图片的尺寸
images_name = os.listdir(img_root)
images_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
#for(i=1;i<471;++i)
for i in range(1,480):
    frame = cv2.imread('{}/{}'.format(img_root, images_name[i]))
    videoWriter.write(frame)
videoWriter.release()