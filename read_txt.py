import os
import cv2
import re
def sort_dir(x):
    num = x.split('/')[-1].split('.')[0]
    return int(num)

def read_txt(video_filename, result_path=None):

    if result_path is None:
        result_path = os.path.splitext(video_filename)[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # capture the video
    vid_cap = cv2.VideoCapture(video_filename)
    total_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(video_filename.split('/')[-1].split('.')[0])
    # start processing
    print("There are {} frames in the video {}".format(total_frame, video_filename))
    print(result_path.split('/')[-1])

    video_frame = []
    with open('/private/poweryin/MCPRL-master/py-faster-rcnn/txtfile/test_bg_all.txt', 'a') as f:
        for i in range(total_frame):

            # read a frame
            success, image = vid_cap.read()

            # save as a JPEG file
            #cv2.imwrite("{}/{}.jpg".format(result_path, i+1), image)

            f.write("{}/{}.jpg".format(result_path.split('/')[-1], i+1))
            f.write('\n')
            # exit if Escape is hit
            if cv2.waitKey(10) == 27:
                break



            # print the progress bar
            # if i % 10 == 0:
            #     print("Done {}/{}".format(i, total_frame))
            #
            # pass
    f.close()
    cv2.destroyAllWindows()
    vid_cap.release()
    print("Done 100%")

    return video_frame


if __name__ == '__main__':
    # path = "./videos/"
    path="/private/poweryin/anomal_data/aic19-track3-test-data"
    path_videos = "/private/poweryin/anomal_data/all_imgs/frames/test"
    dirName = os.listdir(path)
    dirName.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    print(dirName)
    # new_dir = sorted(dirName, key=lambda i: int(re.match(r'(\d+)', i).group()))
    for temp in dirName:
        temp1 = "{}/{}".format(path,temp)
        sort_dir(temp1)
        temp2 = "{}/{}".format(path_videos, temp)
        read_txt(temp1,temp2.split('.',1)[0])

    # video_to_images(path, path_videos)
    # pass