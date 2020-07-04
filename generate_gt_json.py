# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
import scipy.io as scio
import os
import re
import argparse
import json
from pathlib import Path
import os
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import tensorflow as tf
import math
import json
def get_result_():
    def tf_confusion_metrics(predict, real, session, feed_dict):
        print("{:15s} {:15s} {:15s} {:15s}".format("abnormalEvent", "TPR", "FPR", "MAR"))
        predictions = tf.argmax(predict, 1)
        actuals = tf.argmax(real, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)
        video_cls='1'
        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        tn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )
        classes = video_cls
        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )

        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )
        tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op], feed_dict)

        tpr = float(tp) / (float(tp) + float(fn))
        fpr = float(fp) / (float(fp) + float(tn))
        fnr = float(fn) / (float(tp) + float(fn))

        accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
        recall = tpr
        precision = float(tp) / (float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
        print("{} {} {} {}".format(classes, recall, fpr, fnr))


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



def video_kmeans():
    d = {}
    video_path="/home/z840/dataset/UCF_Crimes/test_frames_tufa_crop/"
    All_Folder = os.listdir(video_path)
    All_Folder.sort()
    # All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    with open("/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/gt.json", 'w') as f:
        for video_name in All_Folder:
            filepath = os.path.join(video_path, video_name)
            # read binary data
            feature_folder=os.listdir(filepath)
            frame_lenth=len(feature_folder)
            feature_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
            # for id in feature_folder:
            #     filepath_ = os.path.join(filepath, id)
            txt_list_file = "/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/tufa_anno_gt_json.txt"
            txt_list = open(txt_list_file).readlines()
            tmp=len(txt_list)
            for item in txt_list:
                if video_name in item:
                    ano_class=item.split()[1]
                    if ano_class=='Arson':
                        Label=0
                    if ano_class=='Explosion':
                        Label = 1
                    if ano_class=='RoadAccident':
                        Label=2
                    sum_clip=math.ceil(frame_lenth/16)
                    frame_start = int(item.split()[2])
                    frame_end = int(item.split()[3])
                    if frame_start%16!=0:
                        gt1=[3]*math.ceil(frame_start/16)
                    # ano_gt=[frame_start,frame_end]
                    # if frame_start>1:
                    #     nor_gt1=[1,frame_start-1]
                    # if frame_end<frame_lenth:
                    #     nor_gt2=[frame_end+1,frame_lenth]
                    if frame_end>frame_lenth:
                        frame_end=frame_lenth
                    if frame_end==frame_lenth:
                        gt2=[Label]*(sum_clip-math.ceil(frame_start/16))
                        anno_gt=gt1+gt2
                    if frame_end<frame_lenth and frame_end%16!=0:
                        gt3=[Label]*(math.ceil(frame_end/16)-math.ceil(frame_start/16))
                        gt4=[3]*(sum_clip-math.ceil(frame_end/16))
                        anno_gt=gt1+gt3+gt4
                    d[video_name] = anno_gt
                    if video_name == 'RoadAccident124_x264':
                        print('find')
        json.dump(d,f)
                    # write_str = video_name + ' ' +str(anno_gt)+ '\n'
                    # f.write(write_str)
    f.close()
if __name__ == '__main__':

    video_kmeans()
    get_result_()







