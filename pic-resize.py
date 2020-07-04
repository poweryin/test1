import cv2
image=cv2.imread('/home/z840/Downloads/UMN/wave/1043.jpg')
a = image.shape
print(a)
p1=cv2.resize(image,(int(a[1]/2),int(a[0]/2)),
               interpolation=cv2.INTER_CUBIC)

cv2.imshow('resize', p1)
cv2.waitKey()
cv2.destroyAllWindows()