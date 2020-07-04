import re
import os
import fnmatch
import numpy as np
path="/home/z840/dataset/UCF_Crimes/test_frames_tufa_crop"

out_path ='/home/z840/private/yin/train-class/Normal/'
video_path=os.listdir(path)
nb_frames=16

def count_files(directory, prefix_list):
    lst = os.listdir(directory)
    cnt_list = [len(fnmatch.filter(lst, '*' + x)) for x in prefix_list]

    return cnt_list
for video_name  in video_path:
    frame_path=os.path.join(path,video_name)
    all_cnt = count_files(frame_path, ('.jpg'))
    total_frames = all_cnt[-1]
    print('Total frames: %d' % total_frames)
    valid_frames = total_frames // nb_frames * nb_frames
    move_frames=16-(total_frames-valid_frames)
    print('Total validated frames: %d' % valid_frames)
    frame_id = os.listdir(frame_path)
    frame_id.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for i in range(int(valid_frames // nb_frames)+1):
        clip = np.array([resize(io.imread(os.path.join(video_path, '{}.jpg'.format(j))),
                                output_shape=(resize_w, resize_h), preserve_range=True) for j in
                         range(i * nb_frames + 1, (i + 1) * nb_frames + 1)])
        # clip = np.array([resize(video.get_data(j), output_shape=(resize_w, resize_h), preserve_range=True) for j in range(i * nb_frames, (i+1) * nb_frames)])
        clip = clip[:, index_w: index_w + crop_w, index_h: index_h + crop_h, :]
        clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
        clip = Variable(clip).cuda() if RUN_GPU else Variable(clip)
        clip = clip.resize(1, 3, nb_frames, crop_w, crop_h)
        # print('clip', clip)
        _, clip_output = net(clip, EXTRACTED_LAYER)


if __name__ == '__main__':
    opt = get_opt()

    opt.ngpus_per_node = torch.cuda.device_count()
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    model_path = "./MSRes35_3D_checkpoint/MSRes35_20000.pth"
    main_worker(-1, opt, model_path)