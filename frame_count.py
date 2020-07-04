import os
import scipy.io as scio
VIDEO_DIR ="/data/UCF_Crimes/test_frames"
temp_path ="/data/UCF_Crimes/frame_count/"
video_list_file = "/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"
# video_list_file = "/home/z840/poweryin/AnomalyDetection_MIL/Anomaly_Train.txt"
video_list = open(video_list_file).readlines()
video_list = [item.strip() for item in video_list]
print('video_list', video_list)
for video_name in video_list:
    video_name=video_name.split('/')[-1].split('.')[0]
    frame_path = os.path.join(VIDEO_DIR, video_name)
    frame_length=len(os.listdir(frame_path))
    print(frame_length)
    save_fn = os.path.join(temp_path,'{}.mat'.format(video_name))
    save_array_x = frame_length
    scio.savemat(save_fn, {'frame_length': save_array_x})
