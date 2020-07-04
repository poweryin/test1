import os
import sys
hdf5_list="/data/UCF_Crimes/Videos/output_train_frm/"
txt_list_file="/data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"
txt_list=open(txt_list_file).readlines()

for item in txt_list:
    video_name=item.split("/",1)[1].split(".",1)[0]
    video_path = os.path.join(hdf5_list,video_name)

    if os.path.exists(video_path):
        os.remove(video_path)
    else:
        continue


