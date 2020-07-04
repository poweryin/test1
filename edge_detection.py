# coding:utf-8
import cv2
# Sobel边缘检测算子
# img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/Arson016_x264/1069.jpg', 0)
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
# Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
# Scale_absY = cv2.convertScaleAbs(y)
# result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
# cv2.imshow('img', img)
# cv2.imshow('Scale_absX', Scale_absX)
# cv2.imshow('Scale_absY', Scale_absY)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# canny
# lowThreshold = 0
# max_lowThreshold = 100
# ratio = 3
# kernel_size = 3
#
# img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/Arson016_x264/1069.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# cv2.namedWindow('canny demo')
#
# cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
#
# CannyThreshold(0)  # initialization
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

# canny算子
img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/Arson016_x264/1069.jpg', 0)
blur = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
canny = cv2.Canny(blur, 50, 150)  # 50是最小阈值,150是最大阈值
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()