import os
import cv2


def video_to_images(video_filename, result_path=None):

    if result_path is None:
        result_path = os.path.splitext(video_filename)[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # capture the video
    vid_cap = cv2.VideoCapture(video_filename)
    total_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # start processing
    print("There are {} frames in the video {}".format(total_frame, video_filename))

    video_frame = []

    for i in range(total_frame):

        # read a frame
        success, image = vid_cap.read()

        # save as a JPEG file
        cv2.imwrite("{}/{}.jpg".format(result_path, i+1), image)

        # exit if Escape is hit
        if cv2.waitKey(10) == 27:
            break

        # print the progress bar
        if i % 10 == 0:
            print("Done {}/{}".format(i, total_frame))

        pass

    cv2.destroyAllWindows()
    vid_cap.release()
    print("Done 100%")

    return video_frame


if __name__ == '__main__':

    # path = "./videos/"
    # path="/data/show/test"
    # path_videos = "/data/show/picture"
    # path_videos = "/home/z840/Downloads/UMN/img"
    # path= "/home/z840/Downloads/UMN/video"
    path_videos = "/home/z840/private/yin/train-class-unrename/Explosion/46"
    path= "/home/z840/dataset/DATASET/UCF-Anomaly-Detection-Dataset/UCF_Crimes/Videos/Explosion/test"
    dirName = os.listdir(path)
    for temp in dirName:
        temp1 = "{}/{}".format(path,temp)
        temp2 = "{}/{}".format(path_videos, temp)
        video_to_images(temp1,temp2.split('.',1)[0])

    # video_to_images(path, path_videos)
    # pass