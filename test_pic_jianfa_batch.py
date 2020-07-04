import cv2
import numpy as np
import os
import re

def pic_sub(dest,s1,s2):
    for x in range(dest.shape[0]):
        for y in range(dest.shape[1]):
            if(s2[x,y] > s1[x,y]):
                dest[x,y] = s2[x,y] - s1[x,y]
            else:
                dest[x,y] = s1[x,y] - s2[x,y]

            if(dest[x,y] < threshod):
                dest[x,y] = 0
            else:
                dest[x,y] = 255




bg_path = "/home/z840/dataset/UCF_Crimes/crop_out_30/bg/"
res_path = "/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/"
result_path="/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/"
# tmp=int(res_path.split('/')[-1].split('.')[0])
threshod= 150
All_Folder = os.listdir(bg_path)
All_Folder.sort()
# All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
for video_name in All_Folder:

    filepath = os.path.join(bg_path, video_name)
    filepath_=os.path.join(res_path,video_name)
    diffpath_=os.path.join(result_path,video_name)
    if not os.path.exists(diffpath_):
        os.makedirs(diffpath_)
    # read binary data
    bg_folder=os.listdir(filepath)
    lenth=len(bg_folder)
    bg_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for id in bg_folder:
        id_=str(int(id.split('.')[0]))+'.jpg'
        bgpath_ = os.path.join(filepath, id)
        respath_=os.path.join(filepath_, id_)

        s1 = cv2.imread(bgpath_, 0)
        s2 = cv2.imread(respath_, 0)
        emptyimg = np.zeros(s1.shape, np.uint8)

        pic_sub(emptyimg,s1,s2)

        # cv2.namedWindow("s1")
        # cv2.namedWindow("s2")
        # cv2.namedWindow("result")
        #
        # cv2.imshow("s1",s1)
        # cv2.imshow("s2",s2)
        # cv2.imshow("result",emptyimg)
        tmp_path=os.path.join(diffpath_, "{}".format(id_))
        cv2.imwrite(os.path.join(diffpath_,"{}".format(id_)), emptyimg)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()