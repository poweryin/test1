# coding:utf-8
#!!!!!!!!!此代码必须用python2运行
import os
import cv2
import re
def images_to_video(save_path,video_folder, rep=5, result_filename=None):

    # 生成一组视频
    #result_filename = "{}.mp4".format(result_filename)
    #生成一个视频
    if result_filename is None:
        result_filename = "{}".format(save_path)
    #path = "./**"  生成视频路径   ./**.avi
    #path = "./**/" 生成视频路径   ./**/.avi(这种情况下会找不到.avi文件，它会隐藏掉，所以一般
    # 要在终端输入：ls -a，查看文件夹里所有的文件。再rm .avi  删除掉隐藏文件)

    images_name = os.listdir(video_folder)
    images_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    # images_name.sort(key=lambda i: int(re.match(r'(\d+)', i[3:]).group()))
    # images_name = {int(os.path.splitext(f)[0]): os.path.join(video_folder, f) for f in os.listdir(video_folder)}
    # images_name= sorted(images_name.items(),key=lambda x:x[0])
    # print(images_name)
    # images_name = {int(os.path.splitext(f)[0]): os.path.join(video_folder, f) for f in os.listdir(video_folder)}
    # images_name.sort()
    # read the first frame and find the height, width and layers of all the images
    img = cv2.imread('{}/{}'.format(video_folder, images_name[0]))
    height, width, layers = img.shape


    # initiate the video with width, height and fps = 25
    #four_cc = cv2.VideoWriter_fourcc(*"XVID")  # avi
    four_cc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4
    video = cv2.VideoWriter(result_filename, four_cc, 30, (width, height))

    # for i in range(0, len(images_name)):
    for i in range(0, len(images_name)):
        for j in range(rep):
            t='{}/{}'.format(video_folder, images_name[i])
            img = cv2.imread('{}/{}'.format(video_folder, images_name[i]))
            video.write(img)

        # print the progress bar
        if i % 100 == 0:
            print("Done {}%".format((i*100)/len(images_name)))

    cv2.destroyAllWindows()
    video.release()
    print("Done!")

    return None

if __name__ == '__main__':
    images_path = "/home/z840/dataset/UCF_Crimes/crop_out_30/bg/Arson016_x264"
    save_path1 = "/home/z840/dataset/UCF_Crimes/crop_out_30/bg_video"
    save_name = "Arson016_x264.mp4"

    save = os.path.join(save_path1, save_name)
    # 1S播放两次图片
    images_to_video(save,images_path, 1)


    #path = "/home/z840/dataset/UCF_Crimes/crop_out_30/test_add"
    #path_video = "/home/z840/dataset/UCF_Crimes/crop_out_30/test_video"
    # # path = "/home/z840/dataset/UCF_Crimes/crop_out_30/bg"
    # # path_video = "/home/z840/dataset/UCF_Crimes/crop_out_30/bg_video"
    #dirName = os.listdir(path)
    #for temp in dirName:

    #    temp1 = "{}/{}".format(path, temp)
    #    temp2 = "{}/{}".format(path_video, temp)
     #   images_to_video(temp2,temp1,1)
    #pass
