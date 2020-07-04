import cv2
import tensorflow as tf
from PIL import Image


def showimage_variable_opencv(filename):
    image = cv2.imread(filename)

    #    Create a Tensorflow variable
    image_tensor = tf.Variable(image, name='image')

    with tf.Session() as sess:
        #        image_flap = tf.transpose(image_tensor, perm = [1,0,2])
        sess.run(tf.global_variables_initializer())
        result = sess.run(image_tensor)

    cv2.imshow('result', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    path = "/home/z840/poweryin/test1/023.jpg"
     #showimage_variable_opencv(path)
    cv2.namedWindow("image",0)
    img = cv2.imread(path)
    cv2.imshow("image",img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
