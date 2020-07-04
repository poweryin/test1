import os
path='/home/z840/private/yin/train-class/Normal/test/'
out_path ='/home/z840/private/yin/train-class/Normal/test/'
video_path=os.listdir(path)
for video_id  in video_path:
    src=os.path.join(path,video_id)
    video=video_id.split('.')[0]
    dst=os.path.join(out_path,video)
    os.rename(src, dst)