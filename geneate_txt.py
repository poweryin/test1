import os

data_path='/home/z840/dataset/UCF_Crimes/test_frames_tufa_crop'
# data_path='/home/z840/private/yin/train-class'
save_path="/home/z840/disktend/oldhome/poweryin/test1/test.txt"
class_dict = {'Anomaly':0,'Normal':1}
with open(save_path,'w') as f:
    for file in os.listdir(data_path):
        class_int = class_dict[file]
        for dir_ in os.listdir(os.path.join(data_path,file)):
            # write_str = file+"/"+dir_ + ' '+str(class_int)+'\n'
            write_str = file + "/" + dir_ + ' ' + '\n'
            f.write(write_str)
f.close()