import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import average, linalg, dot
from sklearn import metrics as mr
from scipy.misc import imread
import numpy as np

# zhifangtu
# img = cv2.imread('/home/z840/dataset/UCF_Crimes/crop_out_30/residuals/Arson016_x264/1069.jpg',0)
# plt.hist(img.ravel(),256,[0,256])
# plt.show()

# cosin distnce
def get_thumbnail(image, size=(320, 240), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res


image1 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/666_6.jpg')
image2 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/788_6.jpg')
image3 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/878_6.jpg')
image4 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/911_6.jpg')
image5 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/1079_6.jpg')
image6 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/1153_6.jpg')
image7 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/1179_6.jpg')
image8 = Image.open('/home/z840/dataset/UCF_Crimes/crop_test_out_30/6_/1322_6.jpg')
cosin = image_similarity_vectors_via_numpy(image1, image2)
cosin1 = image_similarity_vectors_via_numpy(image1, image3)
cosin2 = image_similarity_vectors_via_numpy(image1, image7)
cosin3 = image_similarity_vectors_via_numpy(image3, image4)
cosin4 = image_similarity_vectors_via_numpy(image3, image5)
cosin5 = image_similarity_vectors_via_numpy(image3, image6)
print(cosin)
print(cosin1)
print(cosin2)
print(cosin3)
print(cosin4)
print(cosin5)
'''
0.9948215055589822
0.9810545908750539
0.9694126785967297
0.9829294322547577
0.9758853791427564
0.9602154247604128
'''

