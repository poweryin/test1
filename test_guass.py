# coding:utf-8
# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
filename="/home/z840/dataset/UCF_Crimes/test_video_tufa/Arson016_x264.mp4"
out_path="/home/z840/dataset/UCF_Crimes/output/guass/frame"
# 第一步：使用cv2.VideoCapture读取视频
cap = cv2.VideoCapture(filename)
video_name=filename.split("/")[-1].split(".")[0]
# 视频文件输出参数设置
out_fps = 30.0  # 输出文件的帧率
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4
out1 = cv2.VideoWriter(os.path.join('/home/z840/dataset/UCF_Crimes/output/guass', "{}_all.mp4".format(video_name)),
                       fourcc, out_fps, (240, 320))
out2 = cv2.VideoWriter(os.path.join('/home/z840/dataset/UCF_Crimes/output/guass', "{}_fore.mp4".format(video_name)),
                       fourcc, out_fps, (240, 320))
out3 = cv2.VideoWriter(os.path.join('/home/z840/dataset/UCF_Crimes/output/guass', "{}_bg.mp4".format(video_name)),
                       fourcc, out_fps, (240, 320))
# 第二步：cv2.getStructuringElement构造形态学使用的kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 第三步：构造高斯混合模型
knn_model=cv2.BackgroundSubtractorKNN()
model = cv2.createBackgroundSubtractorMOG2()

while(True):
    # 第四步：读取视频中的图片，并使用高斯模型进行拟合
    ret, frame = cap.read()
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    fgmk = model.apply(frame)
    # knn_fgmk=knn_model.apply(frame)
    bg_img = model.getBackgroundImage()
    # knn_bg=knn_model.getBackgroundImage()
    # 第五步：使用形态学的开运算做背景的去除
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    # 第六步：cv2.findContours计算fgmk的轮廓
    contours = cv2.findContours(fgmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for c in contours:
        # 第七步：进行人的轮廓判断，使用周长，符合条件的画出外接矩阵的方格
        length = cv2.arcLength(c, True)

        if length > 188:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 第八步：进行图片的展示
    cv2.imshow('fgmk', fgmk)
    cv2.imshow('frame', frame)
    cv2.imshow('bg_img', bg_img)
    # cv2.imshow('knn_bg',knn_bg)
    # 保存视频
    out1.write(frame)
    out2.write(fgmk)
    out3.write(bg_img)
    if cv2.waitKey(150) & 0xff == 27:
        break

out1.release()
out2.release()
out3.release()
cap.release()
cv2.destroyAllWindows()

