import os
import cv2

if __name__ == "__main__":
    path = '/home/z840/poweryin/ano_pred_cvpr2018-master/Data/shanghaitech/testing/frames/02_0161'
    images = os.listdir(path)

    for image in images:
        img = cv2.imread(os.path.join(path, image))
        print(img.shape)