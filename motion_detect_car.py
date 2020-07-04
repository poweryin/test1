#coding:utf-8
import cv2
import numpy as np
import os
# out_path="/home/z840/dataset/UCF_Crimes/crop_out_30/silance_test/bg_test/pic_box"
out_path="/home/z840/dataset/UCF_Crimes/test_demo/Anomal_box_detect/bg_box"
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
camera = cv2.VideoCapture("/home/z840/dataset/UCF_Crimes/test_demo/Anomal_box_detect/bg_video/RoadAccidents132_x264.mp4")
i=0
while True:
    ret, frame = camera.read()
    fgmask = bs.apply(frame)
    fg2 = fgmask.copy()
    # th = cv2.threshold(fg2, 244, 255, cv2.THRESH_BINARY)[1]
    th = cv2.threshold(fg2, 245, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # 对于矩形区域，只显示>给定阈值的轮廓
        # if cv2.contourArea(c) > 100:
        # 对于矩形区域，只显示>给定阈值的轮廓
        if cv2.contourArea(c) >200:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    i+=1
    cv2.imshow("mog", fgmask)
    cv2.imshow("thresh", th)
    cv2.imshow("detection", frame)
    cv2.imwrite(os.path.join(out_path,"{}.jpg".format(i)), frame)

    if cv2.waitKey(24) & 0xff == 27:
        break
camera.release()
cv2.destroyAllWindows()