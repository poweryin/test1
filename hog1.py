#coding:utf-8
from skimage import feature, exposure
import cv2


image = cv2.imread('/home/z840/Downloads/UMN/wave/1837.jpg')
fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(20, 20),
                    cells_per_block=(2, 2), visualize=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

cv2.imshow('img', image)
cv2.imshow('hog', hog_image_rescaled)
cv2.waitKey(0)==ord('q')