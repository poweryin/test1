import cv2
import matplotlib.pyplot as plt

import numpy as np

img = cv2.imread("/home/z840/dataset/UCF_Crimes/crop_test_out_30/simi/Arson016_x264/1000.jpg")
# img = img[...,::-1]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# step1. gauss blur
img = cv2.GaussianBlur(img,(5,5), 0)

# step2. LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

# step3. compute distance to mean lab value
l_mean = np.average(lab[0])
a_mean = np.average(lab[1])
b_mean = np.average(lab[2])
lab = np.square(lab - np.array([l_mean, a_mean, b_mean]))
lab = np.sum(lab,axis=2)
lab = lab/np.max(lab)

plt.imshow(lab, cmap='gray')
plt.show()
