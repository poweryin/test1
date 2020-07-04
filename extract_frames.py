# coding: utf-8
#将视频提取成图片
import os

# read video list from the folder
# video_list = os.listdir(VIDEO_DIR)

# read video list from the txt list
# video_list_file = "/data/UCF_Crimes/Videos/name_list.txt"
video_list_file = "/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"
video_list = open(video_list_file).readlines()
video_list = [item.strip() for item in video_list]
print('video_list', video_list)
VIDEO_DIR="/home/z840/dataset/DATASET/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos/"
gpu_id = 0

    # if not os.path.isdir(OUTPUT_DIR):
    #     os.mkdir(OUTPUT_DIR)
    # f = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'w')

    # current location
temp_path = "/home/z840/private/yin/train-add/"
if not os.path.exists(temp_path):
    os.mkdir(temp_path)


for video_name in video_list:
    video_path = os.path.join(VIDEO_DIR, video_name)
    print('video_path', video_path)
    frame_path = os.path.join(temp_path, video_name.split('/',1)[1])
    frame_path=frame_path.split('.',1)[0]
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    print('Extracting video frames ...')
        # using ffmpeg to extract video frames into a temporary folder
        # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
    os.system('ffmpeg -i ' + video_path + ' -q:v 2   ' + frame_path + '/%6d.jpg')







