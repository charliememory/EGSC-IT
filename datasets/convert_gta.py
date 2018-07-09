"""Downloads and converts Market1501 data to TFRecords of TF-Example protos.

This module downloads the Market1501 data, uncompresses it, reads the files
that make up the Market1501 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

#from datasets import dataset_utils
import dataset_utils
import numpy as np
import pickle
import pdb
import glob
import cv2 as cv 
import scipy.misc
import shutil
import time

# Seed for repeatability.
random.seed(1)

# The number of shards per dataset split.
_NUM_SHARDS = 1

_IMG_PATTERN_list = ['*.png','*.jpg']

_IMG_HEIGHT = 256
_IMG_WEIGHT = 512

# _IMG_HEIGHT = 512
# _IMG_WEIGHT = 1024


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB png data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        # self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
        self._decode_png = tf.image.decode_image(self._decode_png_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
                                         feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_image_file_list(folder_path):
    filelist = sorted(os.listdir(folder_path))
    # Remove non-jpg non-png files
    valid_filelist = []
    for i in xrange(0, len(filelist)):
        if filelist[i].endswith('.jpg') or filelist[i].endswith('.png'):
            valid_filelist.append(filelist[i])
    return valid_filelist

def _get_file_path_list(folder_path):
    pathlist = []
    for _IMG_PATTERN in _IMG_PATTERN_list:
        pathlist += glob.glob(os.path.join(folder_path, _IMG_PATTERN))
    pathlist = sorted(pathlist)
    # Remove non-jpg non-png non-mat files
    valid_pathlist = []
    for i in xrange(0, len(pathlist)):
        if pathlist[i].endswith('.jpg') or pathlist[i].endswith('.png') or pathlist[i].endswith('.mat'):
            valid_pathlist.append(pathlist[i])

    return valid_pathlist


def _get_dataset_filename(dataset_dir, split_name, shard_id, save_sub_dir):
    output_filename = 'gta_%s_%05d-of-%05d.tfrecord' % (
            split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, save_sub_dir, output_filename)

##################### one_pair_rec ###############
import scipy.io
import scipy.stats
import skimage.morphology
from skimage.morphology import square, dilation, erosion
from PIL import Image

def _img_resize_flip(path, do_fliplr, tmp_dir, image_width=512, image_height=256):
    # img = imread(path)
    img = scipy.misc.imread(path)
    if do_fliplr:
        img = np.fliplr(img)
    img = scipy.misc.imresize(img, [image_height, image_width])
    path_new = os.path.join(tmp_dir, path.split('/')[-2]+'.'+path.split('/')[-1].split('.')[-1])
    scipy.misc.imsave(path_new, img)
    return path_new

def _img_fliplr_oneHot_zoom(path, do_fliplr, image_width=512, image_height=256, num_seg_masks=8):
    img = scipy.misc.imread(path)
    if do_fliplr:
        img = np.fliplr(img)
    img_nd = one_hot(img.astype(np.int64), num_seg_masks)
    img_nd = scipy.ndimage.interpolation.zoom(img_nd, (image_height/8.0/img_nd.shape[0], image_width/8.0/img_nd.shape[1],1), mode="nearest")
    return img_nd

def one_hot(image_in, num_classes=8):
    hot = np.zeros((image_in.shape[0], image_in.shape[1], num_classes))
    layer_idx = np.arange(image_in.shape[0]).reshape(image_in.shape[0], 1)
    component_idx = np.tile(np.arange(image_in.shape[1]), (image_in.shape[0], 1))
    hot[layer_idx, component_idx, image_in] = 1
    return hot.astype(image_in.dtype)


def _format_data(sess, image_reader, idx, tmp_dir, pathlist_A, pathlist_B, 
                pathlist_A_seg, pathlist_B_seg, pathlist_A_seg_class, pathlist_B_seg_class, B_seg_valid_list):
    ## Resize and random flip
    # if np.random.rand()>0.5:
    #     IMG_FLIP = True
    # else:
    #     IMG_FLIP = False
    IMG_FLIP = False
    path_A = _img_resize_flip(pathlist_A[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    path_B = _img_resize_flip(pathlist_B[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    path_A_seg = _img_resize_flip(pathlist_A_seg[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    path_B_seg = _img_resize_flip(pathlist_B_seg[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    # pdb.set_trace()
    path_A_seg_class = _img_resize_flip(pathlist_A_seg_class[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    path_B_seg_class = _img_resize_flip(pathlist_B_seg_class[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    # nd_A_seg_class = _img_fliplr_oneHot_zoom(pathlist_A_seg_class[idx], IMG_FLIP, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    # nd_B_seg_class = _img_fliplr_oneHot_zoom(pathlist_B_seg_class[idx], IMG_FLIP, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    
    image_raw_A = tf.gfile.FastGFile(path_A, 'r').read()
    image_raw_B = tf.gfile.FastGFile(path_B, 'r').read()
    image_raw_A_seg = tf.gfile.FastGFile(path_A_seg, 'r').read()
    image_raw_B_seg = tf.gfile.FastGFile(path_B_seg, 'r').read()
    image_raw_A_seg_class = tf.gfile.FastGFile(path_A_seg_class, 'r').read()
    image_raw_B_seg_class = tf.gfile.FastGFile(path_B_seg_class, 'r').read()

    height, width = image_reader.read_image_dims(sess, image_raw_A)
    # pdb.set_trace()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_name_A': dataset_utils.bytes_feature(pathlist_A[idx].split('/')[-1]),
        'image_name_B': dataset_utils.bytes_feature(pathlist_B[idx].split('/')[-1]),
        'image_raw_A': dataset_utils.bytes_feature(image_raw_A),
        'image_raw_B': dataset_utils.bytes_feature(image_raw_B),
        'image_raw_A_seg': dataset_utils.bytes_feature(image_raw_A_seg),
        'image_raw_B_seg': dataset_utils.bytes_feature(image_raw_B_seg),
        'image_raw_A_seg_class': dataset_utils.bytes_feature(image_raw_A_seg_class),
        'image_raw_B_seg_class': dataset_utils.bytes_feature(image_raw_B_seg_class),
        # 'image_raw_A_seg_class': dataset_utils.int64_feature(nd_A_seg_class.reshape(-1).tolist()),
        # 'image_raw_B_seg_class': dataset_utils.int64_feature(nd_B_seg_class.reshape(-1).tolist()),
        'image_format': dataset_utils.bytes_feature('png'),
        'image_height': dataset_utils.int64_feature(height),
        'image_width': dataset_utils.int64_feature(width),
        'A_seg_valid': dataset_utils.int64_feature(1),
        'B_seg_valid': dataset_utils.int64_feature(B_seg_valid_list[idx]),
    }))
    return example

# def _format_data_test(sess, image_reader, idx, tmp_dir, pathlist_A, pathlist_B, 
#                 pathlist_A_seg, pathlist_B_seg):
#     IMG_FLIP = False
#     path_A = _img_resize_flip(pathlist_A[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
#     path_B = _img_resize_flip(pathlist_B[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
#     path_A_seg = _img_resize_flip(pathlist_A_seg[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
#     path_B_seg = _img_resize_flip(pathlist_B_seg[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)

#     image_raw_A = tf.gfile.FastGFile(path_A[idx], 'r').read()
#     image_raw_B = tf.gfile.FastGFile(path_B[idx], 'r').read()
#     image_raw_A_seg = tf.gfile.FastGFile(path_A_seg[idx], 'r').read()
#     image_raw_B_seg = tf.gfile.FastGFile(path_B_seg[idx], 'r').read()

#     height, width = image_reader.read_image_dims(sess, image_raw_A)

#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image_name_A': dataset_utils.bytes_feature(pathlist_A[idx].split('/')[-1]),
#         'image_name_B': dataset_utils.bytes_feature(pathlist_B[idx].split('/')[-1]),
#         'image_raw_A': dataset_utils.bytes_feature(image_raw_A),
#         'image_raw_B': dataset_utils.bytes_feature(image_raw_B),
#         'image_raw_A_seg': dataset_utils.bytes_feature(image_raw_A_seg),
#         'image_raw_B_seg': dataset_utils.bytes_feature(image_raw_B_seg),
#         'image_format': dataset_utils.bytes_feature('png'),
#         'image_height': dataset_utils.int64_feature(height),
#         'image_width': dataset_utils.int64_feature(width),
#         'A_seg_valid': dataset_utils.int64_feature(1),
#         'B_seg_valid': dataset_utils.int64_feature(1),
#     }))
#     return example

def duplicate_list_like(short_list, ref_list):
    dup_times = int(math.ceil(float(len(ref_list))/float(len(short_list))))
    final_list = []
    for i in range(dup_times):
        final_list += short_list
    final_list = final_list[:len(ref_list)]
    return final_list

def cut_list_like(long_list, ref_list):
    long_list = long_list[:len(ref_list)]
    return long_list

def _convert_dataset_one_pair_rec(dataset_dir, split_name, save_sub_dir, tmp_dir, tf_record_pair_num=np.inf, seg_class_type='8catId'):
    """Converts the given pairs to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        pairs: A list of image name pairs.
        labels: label list to indicate positive(1) or negative(0)
        dataset_dir: The directory where the converted datasets are stored.
    """

    assert split_name in ['train', 'trainOnlyLabeled', 'test'] #  'val'
    if 'train'==split_name:
        pathlist_A = _get_file_path_list(os.path.join(dataset_dir, 'trainA'))
        pathlist_B = _get_file_path_list(os.path.join(dataset_dir, 'trainB'))
        pathlist_B_extra = _get_file_path_list(os.path.join(dataset_dir, 'trainB_extra'))
        pathlist_A_seg = _get_file_path_list(os.path.join(dataset_dir, 'trainA_seg'))
        pathlist_B_seg = _get_file_path_list(os.path.join(dataset_dir, 'trainB_seg'))
        pathlist_A_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'trainA_seg_class_'+seg_class_type))
        pathlist_B_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'trainB_seg_class_'+seg_class_type))
        ## Combine trainB with trainB_extra
        B_seg_valid_list = [1]*len(pathlist_B) + [0]*len(pathlist_B_extra)
        pathlist_B = pathlist_B + pathlist_B_extra
        pathlist_B_seg = duplicate_list_like(pathlist_B_seg, pathlist_B)
        pathlist_B_seg_class = duplicate_list_like(pathlist_B_seg_class, pathlist_B)
    elif 'trainOnlyLabeled'==split_name:
        pathlist_A = _get_file_path_list(os.path.join(dataset_dir, 'trainA'))
        pathlist_B = _get_file_path_list(os.path.join(dataset_dir, 'trainB'))
        pathlist_A_seg = _get_file_path_list(os.path.join(dataset_dir, 'trainA_seg'))
        pathlist_B_seg = _get_file_path_list(os.path.join(dataset_dir, 'trainB_seg'))
        pathlist_A_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'trainA_seg_class_'+seg_class_type))
        pathlist_B_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'trainB_seg_class_'+seg_class_type))
        B_seg_valid_list = [1]*len(pathlist_B)
    elif 'test'==split_name:
        pathlist_A = _get_file_path_list(os.path.join(dataset_dir, 'testA'))
        pathlist_B = _get_file_path_list(os.path.join(dataset_dir, 'testB'))
        pathlist_A_seg = _get_file_path_list(os.path.join(dataset_dir, 'testA_seg'))
        pathlist_B_seg = _get_file_path_list(os.path.join(dataset_dir, 'testB_seg'))
        pathlist_A_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'testA_seg_class_'+seg_class_type))
        pathlist_B_seg_class = _get_file_path_list(os.path.join(dataset_dir, 'testB_seg_class_'+seg_class_type))
        B_seg_valid_list = [1]*len(pathlist_B)

    ## Cut A like B
    # combined = list(zip(pathlist_A, pathlist_A_seg, pathlist_A_seg_class))
    # random.shuffle(combined)
    # pathlist_A[:], pathlist_A_seg[:], pathlist_A_seg_class[:] = zip(*combined)       
    # pathlist_A = cut_list_like(pathlist_A, pathlist_B)
    # pathlist_A_seg = cut_list_like(pathlist_A_seg, pathlist_B_seg)
    # pathlist_A_seg_class = cut_list_like(pathlist_A_seg_class, pathlist_B_seg_class)
    ## Duplicate B like A
    pathlist_B = duplicate_list_like(pathlist_B, pathlist_A)
    pathlist_B_seg = duplicate_list_like(pathlist_B_seg, pathlist_A_seg)
    pathlist_B_seg_class = duplicate_list_like(pathlist_B_seg_class, pathlist_A_seg_class)
    B_seg_valid_list = duplicate_list_like(B_seg_valid_list, pathlist_A)
    
    if 'train'==split_name or 'trainOnlyLabeled'==split_name:
        combined = list(zip(pathlist_A, pathlist_B, pathlist_A_seg, pathlist_B_seg, pathlist_A_seg_class, pathlist_B_seg_class, B_seg_valid_list))
        random.shuffle(combined)
        pathlist_A[:], pathlist_B[:], pathlist_A_seg[:], pathlist_B_seg[:], pathlist_A_seg_class[:], pathlist_B_seg_class[:], B_seg_valid_list[:] = zip(*combined)
    # pathlist_A_all, pathlist_B_all, pathlist_A_seg_all, pathlist_B_seg_all, pathlist_A_seg_class_all, pathlist_B_seg_class_all = [], [], [], [], [], []
    # for ii in xrange(augment_ratio):
    #     combined = list(zip(pathlist_A, pathlist_B, pathlist_A_seg, pathlist_B_seg, pathlist_A_seg_class, pathlist_B_seg_class, B_seg_valid_list))
    #     random.shuffle(combined)
    #     pathlist_A[:], pathlist_B[:], pathlist_A_seg[:], pathlist_B_seg[:], pathlist_A_seg_class[:], pathlist_B_seg_class[:], B_seg_valid_list[:] = zip(*combined)
    #     pathlist_A_all = pathlist_A_all + pathlist_A
    #     pathlist_B_all = pathlist_B_all + pathlist_B
    #     pathlist_A_seg_all = pathlist_A_seg_all + pathlist_A_seg
    #     pathlist_B_seg_all = pathlist_B_seg_all + pathlist_B_seg
    #     pathlist_A_seg_class_all = pathlist_A_seg_class_all + pathlist_A_seg_class
    #     pathlist_B_seg_class_all = pathlist_B_seg_class_all + pathlist_B_seg_class
    # elif 'test'==split_name:
    #     pathlist_A = _get_file_path_list(os.path.join(dataset_dir, 'testA'))
    #     pathlist_B = _get_file_path_list(os.path.join(dataset_dir, 'testB'))
    #     pathlist_A_seg = _get_file_path_list(os.path.join(dataset_dir, 'testA_seg'))
    #     pathlist_B_seg = _get_file_path_list(os.path.join(dataset_dir, 'testB_seg'))
    #     pathlist_A_all, pathlist_B_all, pathlist_A_seg_all, pathlist_B_seg_all = [], [], [], []
    #     for ii in xrange(augment_ratio):
    #         combined = list(zip(pathlist_A, pathlist_B, pathlist_A_seg, pathlist_B_seg))
    #         random.shuffle(combined)
    #         pathlist_A[:], pathlist_B[:], pathlist_A_seg[:], pathlist_B_seg[:] = zip(*combined)
    #         pathlist_A_all = pathlist_A_all + pathlist_A
    #         pathlist_B_all = pathlist_B_all + pathlist_B
    #         pathlist_A_seg_all = pathlist_A_seg_all + pathlist_A_seg
    #         pathlist_B_seg_all = pathlist_B_seg_all + pathlist_B_seg

    # num_shards = _NUM_SHARDS
    num_shards = 1
    sample_num = len(pathlist_A) 
    # sample_num = len(pathlist_A_all) 
    num_per_shard = int(math.ceil(sample_num / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(
                        dataset_dir, split_name, shard_id, save_sub_dir)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    cnt = 0

                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, sample_num)

                    for idx in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                idx+1, sample_num, shard_id))
                        sys.stdout.flush()

                        example = _format_data(sess, image_reader, idx, tmp_dir, pathlist_A, pathlist_B, 
                                    pathlist_A_seg, pathlist_B_seg, pathlist_A_seg_class, pathlist_B_seg_class, B_seg_valid_list)
                        # example = _format_data(sess, image_reader, idx, tmp_dir, pathlist_A_all, pathlist_B_all, 
                        #             pathlist_A_seg_all, pathlist_B_seg_all, pathlist_A_seg_class_all, pathlist_B_seg_class_all)
                        # if 'train'==split_name:
                        #     example = _format_data_train(sess, image_reader, idx, tmp_dir, pathlist_A_all, pathlist_B_all, 
                        #                 pathlist_A_seg_all, pathlist_B_seg_all, pathlist_A_seg_class_all, pathlist_B_seg_class_all)
                        # elif 'test'==split_name:
                        #     example = _format_data_test(sess, image_reader, idx, tmp_dir, pathlist_A_all, pathlist_B_all, 
                        #                 pathlist_A_seg_all, pathlist_B_seg_all)

                        if None==example:
                            continue

                        tfrecord_writer.write(example.SerializeToString())
                        cnt += 1
                        if cnt==tf_record_pair_num:
                            break

    sys.stdout.write('\n')
    sys.stdout.flush()
    print('cnt:',cnt)
    with open(os.path.join(dataset_dir, save_sub_dir, 'tf_record_sample_num.txt'),'w') as f:
        f.write('sample_num:%d' % sample_num)


if __name__ == '__main__':
    seg_class_type='8catId'
    # seg_class_type='20trainId'
    # split_name = 'train'
    # split_name = 'trainOnlyLabeled'
    split_name = 'test'
    dataset_dir = '/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/gta25k/'
    save_sub_dir = 'gta25k_city_{}_{}x{}_{}'.format(split_name, _IMG_HEIGHT, _IMG_WEIGHT, seg_class_type)
    # sub_dir_name = '' ## '' or '_day' or '_night' 
    # dataset_dir = '/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/gta25k_bdd%s/'%sub_dir_name
    # save_sub_dir = 'gta25k_bdd{}_{}_{}x{}_{}'.format(sub_dir_name , split_name, _IMG_HEIGHT, _IMG_WEIGHT, seg_class_type)

    if not os.path.exists(os.path.join(dataset_dir,save_sub_dir)):
        os.makedirs(os.path.join(dataset_dir,save_sub_dir))

    tmp_dir = os.path.join('/tmp', str(time.time()))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    _convert_dataset_one_pair_rec(dataset_dir, split_name, save_sub_dir, tmp_dir,
                                    tf_record_pair_num=np.inf, 
                                    seg_class_type=seg_class_type)
