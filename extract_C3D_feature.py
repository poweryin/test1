#!/usr/bin/env python
# picture format:000001.jpg
import re
'''
Extract C3D features as a csv file from a given video, 
'''

import numpy as np
import sys
import os
import subprocess
import array
import cv2

###################################################################
# Point to the C3D directory
# caffe_root = os.path.abspath(os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         '../..'
#         ))
caffe_root="/home/z840/Downloads/C3D/C3D-v1.0"
# GPU to us
gpu_id = 0

# 50 should be good for 6GB VRAM. Decrease as needed
# batch_size = 50
batch_size = 60
###################################################################
def run_C3D_extraction(feature_prototxt, ofile, feature_layer, trained_model):
    ''' Extract C3D features by running caffe binary '''

    almost_infinite_num = 9999999

    extract_bin = os.path.join(
            caffe_root,
            "build/tools/extract_image_features.bin"
            )

    feature_extraction_cmd = [
            extract_bin,
            feature_prototxt,
            trained_model,
            str(gpu_id),
            str(batch_size),
            str(almost_infinite_num),
            ofile,
            feature_layer,
            ]

    print ("[Info] Running C3D feature extraction: \"{}\"".format(
            ' '.join(feature_extraction_cmd)
            ))
    return_code = subprocess.call(feature_extraction_cmd)

    return return_code

def get_features(feature_files, feature_layer):
    ''' From binary feature files, take an average (for multiple clips) '''

    # in case of a single feature_file, force it to a list
    if isinstance(feature_files, basestring):
        feature_files = [feature_files]

    # read each feature, take an an average
    for clip_count, feature_file in enumerate(feature_files):
        print ("clip_count={}, feature_file={}".format(clip_count, feature_file))
        if not os.path.exists(feature_file):
            feature_file += '.' + feature_layer

        if not os.path.exists(feature_file):
            print ("[Error] feature_file={} does not exist!".format(feature_file))
            return None

        # read binary data
        f = open(feature_file, "rb")
        # read all bytes into a string
        s = f.read()
        f.close()
        (n, c, l, h, w) = array.array("i", s[:20])
        feature_vec = np.array(array.array("f", s[20:]))

        if clip_count == 0:
            feature_vec_avg = feature_vec
        else:
            feature_vec_avg += feature_vec

    feature_vec_avg = feature_vec_avg / len(feature_files)

    return feature_vec_avg

def generate_feature_prototxt(out_file, src_file, mean_file=None):
    ''' Generate a model architecture, pointing to the given src_file '''

    # by default, mean file must exist.
    # if for some reason it's missing, get from:
    # https://github.com/facebook/C3D/blob/master/examples/c3d_feature_extraction/sport1m_train16_128_mean.binaryproto?raw=true
    if not mean_file:
        mean_file = os.path.join(
                caffe_root,
                "examples",
                "c3d_feature_extraction",
                "sport1m_train16_128_mean.binaryproto"
                )

    if not os.path.isfile(mean_file):
        print ("[Error] mean cube file={} does not exist.".format(mean_file))
        sys.exit(-8)

    # replace source video clips / mean_file
    prototxt_content = '''
name: "DeepConv3DNet_Sport1M_Val"
layers {{
  name: "data"
  type: VIDEO_DATA
  top: "data"
  top: "label"
  image_data_param {{
    source: "{0}"
    use_image: true
    mean_file: "{1}"
    batch_size: 60
    crop_size: 112
    mirror: false
    show_data: 0
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: false
  }}
}}
# ----------- 1st layer group ---------------
layers {{
  name: "conv1a"
  type: CONVOLUTION3D
  bottom: "data"
  top: "conv1a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 64
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
layers {{
  name: "relu1a"
  type: RELU
  bottom: "conv1a"
  top: "conv1a"
}}
layers {{
  name: "pool1"
  type: POOLING3D
  bottom: "conv1a"
  top: "pool1"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 1
    stride: 2
    temporal_stride: 1
  }}
}}
# ------------- 2nd layer group --------------
layers {{
  name: "conv2a"
  type: CONVOLUTION3D
  bottom: "pool1"
  top: "conv2a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 128
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu2a"
  type: RELU
  bottom: "conv2a"
  top: "conv2a"
}}
layers {{
  name: "pool2"
  type: POOLING3D
  bottom: "conv2a"
  top: "pool2"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# ----------------- 3rd layer group --------------
layers {{
  name: "conv3a"
  type: CONVOLUTION3D
  bottom: "pool2"
  top: "conv3a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 256
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu3a"
  type: RELU
  bottom: "conv3a"
  top: "conv3a"
}}
layers {{
  name: "conv3b"
  type: CONVOLUTION3D
  bottom: "conv3a"
  top: "conv3b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 256
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu3b"
  type: RELU
  bottom: "conv3b"
  top: "conv3b"
}}
layers {{
  name: "pool3"
  type: POOLING3D
  bottom: "conv3b"
  top: "pool3"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# --------- 4th layer group
layers {{
  name: "conv4a"
  type: CONVOLUTION3D
  bottom: "pool3"
  top: "conv4a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu4a"
  type: RELU
  bottom: "conv4a"
  top: "conv4a"
}}
layers {{
  name: "conv4b"
  type: CONVOLUTION3D
  bottom: "conv4a"
  top: "conv4b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu4b"
  type: RELU
  bottom: "conv4b"
  top: "conv4b"
}}
layers {{
  name: "pool4"
  type: POOLING3D
  bottom: "conv4b"
  top: "pool4"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# --------------- 5th layer group --------
layers {{
  name: "conv5a"
  type: CONVOLUTION3D
  bottom: "pool4"
  top: "conv5a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu5a"
  type: RELU
  bottom: "conv5a"
  top: "conv5a"
}}
layers {{
  name: "conv5b"
  type: CONVOLUTION3D
  bottom: "conv5a"
  top: "conv5b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu5b"
  type: RELU
  bottom: "conv5b"
  top: "conv5b"
}}
layers {{
  name: "pool5"
  type: POOLING3D
  bottom: "conv5b"
  top: "pool5"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# ---------------- fc layers -------------
layers {{
  name: "fc6-1"
  type: INNER_PRODUCT
  bottom: "pool5"
  top: "fc6-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 4096
    weight_filler {{
      type: "gaussian"
      std: 0.005
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu6"
  type: RELU
  bottom: "fc6-1"
  top: "fc6-1"
}}
layers {{
  name: "drop6"
  type: DROPOUT
  bottom: "fc6-1"
  top: "fc6-1"
  dropout_param {{
    dropout_ratio: 0.5
  }}
}}
layers {{
  name: "fc7-1"
  type: INNER_PRODUCT
  bottom: "fc6-1"
  top: "fc7-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 4096
    weight_filler {{
    type: "gaussian"
      std: 0.005
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu7"
  type: RELU
  bottom: "fc7-1"
  top: "fc7-1"
}}
layers {{
  name: "drop7"
  type: DROPOUT
  bottom: "fc7-1"
  top: "fc7-1"
  dropout_param {{
    dropout_ratio: 0.5
  }}
}}
layers {{
  name: "fc8-1"
  type: INNER_PRODUCT
  bottom: "fc7-1"
  top: "fc8-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 487
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
layers {{
  name: "prob"
  type: SOFTMAX
  bottom: "fc8-1"
  top: "prob"
}}
layers {{
  name: "accuracy"
  type: ACCURACY
  bottom: "prob"
  bottom: "label"
  top: "accuracy"}}'''.format(src_file, mean_file)

    with open(out_file, 'w') as f:
        f.write(prototxt_content)

    return

def main():
    ''' Extract and save features '''

    # a video is the first argument
    # if missing, use a sample video
#     if len(sys.argv) == 1:
#         print '''Usage: python {} <video file> (<optional output directory>)
# For example, "python {} {}" will extract features from an example image.'''.format(
#         sys.argv[0],
#         sys.argv[0],
#         os.path.join(
#                 caffe_root,
#                 'examples',
#                 'c3d_feature_extraction',
#                 'input',
#                 'avi',
#                 'v_BaseballPitch_g01_c01.avi'
#                 )
#         )
#         sys.exit(-1)

    # trained model (will be downloaded if missing)
    trained_model = os.path.join(
        caffe_root,
        "examples",
        "c3d_feature_extraction",
        "conv3d_deepnetA_sport1m_iter_1900000"
        )
    # check model
    # check_trained_model(trained_model)

    # feature to extract
    feature_layer = 'fc6-1'
    # overwrite feature output?
    force_overwrite = False
    # by default, use 16 frames
    # num_frames_per_clip = 16  # ~0.5 second
    num_frames_per_clip = 16  # ~0.5 second

    # sampling rate (in seconds)
    sample_every_N_sec = 60
    # don't extract beyond this (in seconds)
    max_processing_sec = 599

    # save extracted frames temporarily
    # tmp_dir = '/home/z840/disktend/oldhome/poweryin/ano_pred_cvpr2018-master/Data/shanghaitech/testing/test_frames/'
    # txt_dir="/home/z840/dataset/UCF_Crimes/generate/generate_txt_train/"
    tmp_dir = '/home/z840/dataset/UCF_Crimes/crop_out_30/bg/'
    txt_dir="/home/z840/dataset/UCF_Crimes/generate/generate_txt_train/"
    # generate auxilliary files for C3D feature extraction
    input_file = os.path.join(txt_dir, 'input.txt')
    output_prefix_file = os.path.join(txt_dir, 'output_prefix.txt')
    feature_prototxt = os.path.join(txt_dir, 'feature_extraction.prototxt')
    generate_feature_prototxt(feature_prototxt, input_file)

    video_=os.listdir(tmp_dir)
    video_.sort()
    # c3d_feature_outdir="/home/z840/disktend/oldhome/poweryin/ano_pred_cvpr2018-master/Data/shanghaitech/test_feature"
    c3d_feature_outdir="/home/z840/dataset/UCF_Crimes/crop_out_30/bg_feature"

    for video_name in video_:
        video_name=video_name.split('.')[0]
        frame_dir=os.path.join(tmp_dir,video_name)
        result_dir=os.path.join(c3d_feature_outdir,video_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        video_frame=os.listdir(frame_dir)
        video_frame.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))

        # video_frame.sort(key=lambda i: int(re.match(r'(\d+)', i)))
        # if video_name!=None:
        #     video_frame.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))

        # video_frame.sort()
        # video_frame = list(map(lambda x:"{:06d}.jpg".format(int(x.split('.')[0])),video_frame))
        frame_count=len(video_frame)

        # frame_inc = 16
        frame_inc = 16

        start_frame = 1
        # make sure not to reach the edge of the video
        end_frame = frame_count-frame_count%frame_inc
        start_frames = []
        for frame_index in range(start_frame, end_frame+1, frame_inc):
            #print "[Debug] adding frame_index={}".format(frame_index)
            start_frames.append(frame_index)



    # first, populate input.txt, and output_prefix.txt files
    # each line corresponds to a 16-frame video clip
        f_input = open(input_file, 'w')
        f_output_prefix = open(output_prefix_file, 'w')
        for start_frame in start_frames:
            # output feature file (CSV)
            feature_filename = os.path.join(result_dir,"%d.fc6-1" % start_frame)

            # a dummy label
            dummy_label = 0

            # write "input.txt" with just one clip
            f_input.write("{} {} {}\n".format(frame_dir, start_frame, dummy_label))

            # write "output_prefix.txt" with one clip
            # clip_id = os.path.join(
            #         tmp_dir,
            #         video_id + '_{0:06d}'.format(start_frame)
            #         )
            clip_id = os.path.join(result_dir,"%d" % start_frame)
            f_output_prefix.write("{}\n".format(clip_id))
        f_input.close()
        f_output_prefix.close()

        # second, run C3D extraction (with a batch)
        if os.path.isfile(input_file) and os.path.getsize(input_file):
            return_code = run_C3D_extraction(
                    feature_prototxt,
                    output_prefix_file,
                    feature_layer,
                    trained_model
                    )

            # third, if C3D ran successfully, convert each feature file (binary) to csv
            if return_code == 0:
                for start_frame in start_frames:
                    # output feature file (CSV)
                    feature_filename = os.path.join(result_dir,"%d.fc6-1" % start_frame)

                    if os.path.isfile(feature_filename) and not force_overwrite:
                        print("[Warning] feature was already saved. Skipping this "
                              "video...")
                        continue

                    clip_id = os.path.join(
                            result_dir,
                            "%d" % start_frame
                            )
                    feature = get_features([clip_id], feature_layer)

                    print("[Info] Saving C3D feature as {}".format(
                            feature_filename,
                            ))
                    # save the average feature vector as a CSV
                    np.savetxt(
                            feature_filename,
                            feature[None, :],
                            fmt='%.16f',
                            delimiter=','
                            )
            else:
                print ("[Error] feature extraction failed!")

if __name__ == '__main__':
    main()
