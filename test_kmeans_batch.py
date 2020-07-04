# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import array
import scipy.io as scio
import os
import re
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import tensorflow as tf


# MSC8-3D result
def test_kmeans(*data):
    x,labels_true=data
    clst=cluster.KMeans(n_clusters=2)
    clst.fit(x)
    predicted_labels=clst.predict(x)
    ARI=adjusted_rand_score(labels_true,predicted_labels)
    # print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))
    # print("sum center distance %s"%clst.inertia_)
    return predicted_labels,ARI

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


def video_kmeans():
    predict_path="/home/z840/dataset/UCF_Crimes/test_demo/MSC8-3D/mat_predict_label"
    mat_path="/home/z840/dataset/UCF_Crimes/test_demo/MSC8-3D/mat_feature"
    feature_path="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_feature"
    All_Folder = os.listdir(feature_path)
    All_Folder.sort()
    # All_Folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    for video_name in All_Folder:
        li = []
        filepath = os.path.join(feature_path, video_name)
        # read binary data
        feature_folder=os.listdir(filepath)
        lenth=len(feature_folder)
        feature_folder.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        for id in feature_folder:
            filepath_ = os.path.join(filepath, id)
            f = open(filepath_, "rb")
            # read all bytes into a string
            s = f.read()
            f.close()
            (n, c, l, h, w) = array.array("i", s[:20])
            feature_vec = np.array(array.array("f", s[20:]))
            li.append(feature_vec)
        save_fn = os.path.join(mat_path,'{}.mat'.format(video_name))
        scio.savemat(save_fn, {'idfature': li})
        txt_list_file = "/home/z840/dataset/UCF_Crimes/Anomaly_Detection_splits/tufa-gt.txt"
        txt_list = open(txt_list_file).readlines()
        tmp=len(txt_list)
        for item in txt_list:
            if video_name in item:
                frame_start = int(item.split()[2])
                frame_end = int(item.split()[3])
                frame_start1 = int(item.split()[4])
                frame_end1 = int(item.split()[5])
                bg_start=int(item.split()[6])
                bg_end=int(item.split()[7])
                label = [0] * lenth
                labels_true=np.asarray(label)
                if bg_end==-1:
                    labels_true[bg_start:]=1
                else:
                    labels_true[bg_start:bg_end] = 1
                predict_label,ari = test_kmeans(li,labels_true)
                save_pre = os.path.join(predict_path,'{}.mat'.format(video_name))
                scio.savemat(save_pre, {'id': predict_label})
                print(video_name)
                print(lenth)
                print("labels_true")
                print(labels_true)
                True_1=list(labels_true).index(1)
                print("True_1")
                print(list(labels_true).index(1))
                print((list(labels_true).index(1)) * 16 + 1)
                print(sum(labels_true))
                print(predict_label)
                print(list(predict_label).index(1))
                print((list(predict_label).index(1))*16+1)
                print(sum(predict_label))
                print(sum(labels_true)-sum(predict_label))
                print(ari)

if __name__ == '__main__':
    video_kmeans()
    get_result_()







