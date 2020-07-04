import cv2
import os
import glob as gb
import re
print("开始获取数据...")
datasetDir="/home/z840/dataset/UCF_Crimes/test-add/RoadAccidents132_x264"
out_path="/home/z840/dataset/UCF_Crimes/test-add/Road-addtext"
# img_path = gb.glob(os.path.join(datasetDir, '*.jpg'))
img_path1 = os.listdir(datasetDir)
img_path1.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
# img_path.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
# img_path2=os.path.join(datasetDir,img_path1)
rawImgs = []
opImgs = []
for i, curFrame1 in enumerate(img_path1):
    print("curFrame1: ", curFrame1)
    curFrame=os.path.join(datasetDir,curFrame1)
    img = cv2.imread(curFrame)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    img=cv2.putText(img, "frame:", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255,255), 2)
    img = cv2.putText(img, str(i+1), (75, 20), font, 0.6, (0, 255,255), 2)  # #添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    img = cv2.putText(img, "Anomaly!!!", (130, 20), font, 0.6, (0,255,0), 2)
    img = cv2.putText(img, "GT-Ano", (245, 20), font, 0.6, (255, 0, 255), 2)
    cv2.namedWindow("rawImageFrame", 0)
    cv2.resizeWindow("rawImageFrame", 320, 240)
    cv2.imshow('rawImageFrame', img)
    tmp_path=os.path.join(out_path, "{}.jpg".format(i + 1))
    cv2.imwrite(os.path.join(out_path, "{}.jpg".format(i+1)), img)
    cv2.waitKey(30)
cv2.destroyAllWindows()