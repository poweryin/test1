# coding:utf-8
# 利用opencv获取最大连通区域并去除

# 首先通过findContours函数找到二值图像中的所有边界(这块看需要调节里面的参数)
# 然后通过contourArea函数计算每个边界内的面积
# 最后通过fillConvexPoly函数将面积最大的边界内部涂成背景
import cv2
import numpy as np
import matplotlib.pyplot as plt


#求最大连通域的中心点坐标
def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/test_diff/Arson016_x264/950.jpg')
# img = cv2.imread('/home/z840/Downloads/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#find contours of all the components and holes
gray_temp = gray.copy() #copy the gray image because function
#findContours will change the imput image into another
binary,contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#show the contours of the imput image
cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
plt.figure('original image with contours'), plt.imshow(img, cmap = 'gray')
#find the max area of all the contours and fill it with 0
area = []
for i in range(len(contours)):
    area.append(cv2.contourArea(contours[i]))
max_idx = np.argmax(area)
area_=np.asarray(area)
idx=np.argsort(-area_)
# second_max=idx[1]
##求最大连通域的中心坐标
cnt_centroid = centroid(contours[max_idx])
cv2.circle(contours[max_idx],cnt_centroid,5,[255,0,255],-1)
print("Centroid : " + str(cnt_centroid))

cv2.fillConvexPoly(gray, contours[max_idx], 0)
# cv2.fillConvexPoly(gray,contours[second_max],0)
#show image without max connect components
plt.figure('remove max connect com'), plt.imshow(gray, cmap = 'gray')
plt.show()


