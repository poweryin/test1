
import argparse
import json
from pathlib import Path
import os
import pandas as pd

from util_scripts.utils import get_n_frames


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path, delimiter=' ', header=None)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        slash_rows = data.iloc[i, 0].split('/')
        class_name = slash_rows[0]
        basename = slash_rows[1].split('.')[0] # video name

        keys.append(basename)
        key_labels.append(class_name)

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = key_labels[i]
        database[key]['annotations'] = {'label': label}

    return database


def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.iloc[i, 1])
    return labels


def convert_ucf101_csv_to_json(label_csv_path, train_csv_path, val_csv_path,
                               video_dir_path_train,video_dir_path_test, dst_json_path):
    labels = load_labels(label_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    for k, v in dst_data['database'].items():
        if v['annotations'] is not None:
            label = v['annotations']['label']
        else:
            label = 'test'

        video_dir_path = video_dir_path_train if v['subset']=='training'else video_dir_path_test
        video_path = video_dir_path / label / k
        # n_frames = get_n_frames(video_path)
        # tmp=os.listdir(video_path)
        n_frames=len(os.listdir(video_path))
        v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path',
                        default='/home/z840/Downloads/MSRes35-3D_MSC8-3D/UCF-Crime',
                        type=Path,
                        help=('Directory path including classInd.txt, '
                              'trainlist0-.txt, testlist0-.txt'))
    parser.add_argument('--video_path_train',
                        default='/home/z840/private/yin/train-class',
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('--video_path_test',
                        default='/home/z840/dataset/UCF_Crimes/test_frames_tufa_crop',
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('--dst_path',
                        default='/home/z840/Downloads/MSRes35-3D_MSC8-3D/UCF-Crime',
                        type=Path,
                        help='Directory path of dst json file.')

    args = parser.parse_args()

    for split_index in range(1, 2):
        label_csv_path = args.dir_path / 'classInd.txt'
        train_csv_path = args.dir_path / 'trainlist0{}.txt'.format(split_index)
        val_csv_path = args.dir_path / 'testlist0{}.txt'.format(split_index)
        dst_json_path = args.dst_path / 'ucf_0{}.json'.format(split_index)

        convert_ucf101_csv_to_json(label_csv_path, train_csv_path, val_csv_path,
                                   args.video_path_train,args.video_path_test, dst_json_path)