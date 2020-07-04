
# -*- coding:utf8 -*-
import os
import re
# path = '/data/UCF_Crimes/frames/Fighting006_x264/'
# out_path ='/data/UCF_Crimes/out/'
# filelist = os.listdir(path)
# filelist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
# for item in filelist:
#         #print('item name is ',item)
#         if item.endswith('.jpg'):
#                 name = str(int(item.split('.',1)[0])+276)
#                 src = os.path.join(os.path.abspath(path),item)
#                 dst = os.path.join(out_path,name + '.jpg')
#         try:
#                 os.rename(src,dst)
#                 print('rename from %s to %s'%(src,dst))
#         except:
#                 continue
#先将图片从1开始命名，再将reverse=True倒序，将图片保存到另一个文件夹
#将文件夹中的图片从1.jpg命名为000001.jpg
# path = '/home/z840/private/yin/train-class/RoadAccident/'
# out_path ='/home/z840/private/yin/train-class/RoadAccident/'

path = '/home/z840/private/yin/train-class/Normal/'
# path="/home/z840/dataset/UCF_Crimes/test_frames_add_split/bg/add"
out_path ='/home/z840/private/yin/train-class/Normal/'
video_path=os.listdir(path)
for video_id  in video_path:
    frame_path=os.path.join(path,video_id)
    out_path_=os.path.join(out_path,video_id)
    if not os.path.exists(out_path_):
        os.makedirs(out_path_)
    filelist = os.listdir(frame_path)
    filelist.sort(key=lambda i: int(re.match(r'(\d+)', i).group()),reverse =False)
    i=1
    for item in filelist:
            src = os.path.join(frame_path,item)
            tt=int(item.split('.')[0])

            # 可在此处标注起始帧好从几开始
            #dst = os.path.join(out_path_, "{:04d}.jpg".format(int(item.split('.')[0])))
            dst = os.path.join(out_path_, "{:06d}.jpg".format(i))
            try:
                os.rename(src, dst)
                print('rename from %s to %s' % (src, dst))
            except:
                continue
            i=i+1
        # vidlist = list(map(lambda x:"{:06d}.jpg".format(int(x.split('.')[0])),filelist))

        # dst = os.path.join(out_path,name + '.jpg')
        # for item in filelist:
        #         #print('item name is ',item)
        #         if item.endswith('.jpg'):
        #                 name = str(int(item.split('.',1)[0])+276)
        #                 src = os.path.join(os.path.abspath(path),item)
        #                 dst = os.path.join(out_path,nae + '.jpg')
        #         try:
        #                 os.rename(src,dst)
        #                 print('rename from %s to %s'%(src,dst))
        #         except:
        #                 continue