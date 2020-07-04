import numpy as np
import array
import os
feature_files="/home/z840/dataset/UCF_Crimes/Videos/test_seg/Abuse001_x264"
feature_file="1.fc6-1"
f = open(os.path.join(feature_files, feature_file), "rb")
# read all bytes into a string
s = f.read()
s = s.split()
s=list(map(float,s[20:]))
print(s)
f.close()
t = array.array("f", s[:20])
feature_vec = np.array(array.array("f", s[:20]))
print(t)
print(feature_vec)