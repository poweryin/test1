
import os

VIDEO_DIR ="/data/UCF_Crimes/Videos/"
temp_path ="/private/yin/tmp/"
video_list_file = "/home/z840/poweryin/AnomalyDetection_MIL/Anomaly_Train.txt"
video_list = open(video_list_file).readlines()
video_list = [item.strip() for item in video_list]
print('video_list', video_list)
for video_name in video_list:
    video_path = os.path.join(VIDEO_DIR, video_name)
    print('video_path', video_path)
    frame_path = os.path.join(temp_path, video_name.split('/')[-1])
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    print('Extracting video frames ...')
    # using ffmpeg to extract video frames into a temporary folder
    # example: ffmpeg -i video_validation_0000051.mp4 -q:v 2 -f image2 output/image%5d.jpg
    os.system('ffmpeg -i ' + video_path + ' -q:v 2   ' + frame_path + '/%1d.jpg')

    print('Extracting features ...')
    total_frames = len(os.listdir(frame_path))
    if total_frames == 0:
        print('Fail to extract frames for video: %s' % video_name)
        continue
