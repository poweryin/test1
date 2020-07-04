import cv2

cap = cv2.VideoCapture("/home/z840/dataset/UCF_Crimes/test_video_tufa/RoadAccidents002_x264.mp4")
knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    mog_sub_mask = mog2_sub.apply(frame)
    knn_sub_mask = knn_sub.apply(frame)
    mog_bg = mog2_sub.getBackgroundImage()
    knn_bg= knn_sub.getBackgroundImage()
    cv2.imshow('original', frame)
    cv2.imshow('MOG2', mog_sub_mask)
    cv2.imshow('MOG_bg', mog_bg)
    cv2.imshow('KNN', knn_sub_mask)
    cv2.imshow('KNN_bg', knn_bg)
    key = cv2.waitKey(30) & 0xff
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
