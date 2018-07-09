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

_IMG_HEIGHT = 128
_IMG_WEIGHT = 128


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
    output_filename = 'celeba_%s_%05d-of-%05d.tfrecord' % (
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


def _format_data(sess, image_reader, idx, tmp_dir, pathlist_A, pathlist_B):
    ## Resize and random flip
    # if np.random.rand()>0.5:
    #     IMG_FLIP = True
    # else:
    #     IMG_FLIP = False
    IMG_FLIP = False
    path_A = _img_resize_flip(pathlist_A[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)
    path_B = _img_resize_flip(pathlist_B[idx], IMG_FLIP, tmp_dir, image_width=_IMG_WEIGHT, image_height=_IMG_HEIGHT)

    image_raw_A = tf.gfile.FastGFile(path_A, 'r').read()
    image_raw_B = tf.gfile.FastGFile(path_B, 'r').read()

    height, width = image_reader.read_image_dims(sess, image_raw_A)
    # pdb.set_trace()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_name_A': dataset_utils.bytes_feature(pathlist_A[idx].split('/')[-1]),
        'image_name_B': dataset_utils.bytes_feature(pathlist_B[idx].split('/')[-1]),
        'image_raw_A': dataset_utils.bytes_feature(image_raw_A),
        'image_raw_B': dataset_utils.bytes_feature(image_raw_B),
        'image_format': dataset_utils.bytes_feature('png'),
        'image_height': dataset_utils.int64_feature(height),
        'image_width': dataset_utils.int64_feature(width),
    }))
    return example

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

def _convert_dataset_one_pair_rec(dataset_dir, split_name, save_sub_dir, tmp_dir, tf_record_pair_num=np.inf):
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
    elif 'test'==split_name:
        pathlist_A = _get_file_path_list(os.path.join(dataset_dir, 'testA'))
        pathlist_B = _get_file_path_list(os.path.join(dataset_dir, 'testB'))

    ## Duplicate B like A
    if len(pathlist_A)>len(pathlist_B):
        pathlist_B = duplicate_list_like(pathlist_B, pathlist_A)
    else:
        pathlist_A = duplicate_list_like(pathlist_A, pathlist_B)

    if 'train'==split_name:
        combined = list(zip(pathlist_A, pathlist_B))
        random.shuffle(combined)
        pathlist_A[:], pathlist_B[:] = zip(*combined)
    # pdb.set_trace()
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

                        example = _format_data(sess, image_reader, idx, tmp_dir, pathlist_A, pathlist_B)

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
    dataset_dir = int(sys.argv[1])
    IsTrain = int(sys.argv[2])

    if IsTrain:
        split_name = 'train'
    else:
        split_name = 'test'
    save_sub_dir = 'celebaMaleFemale_{}_{}x{}'.format(split_name, _IMG_HEIGHT, _IMG_WEIGHT)

    if not os.path.exists(os.path.join(dataset_dir,save_sub_dir)):
        os.makedirs(os.path.join(dataset_dir,save_sub_dir))

    tmp_dir = os.path.join('/tmp', str(time.time()))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    _convert_dataset_one_pair_rec(dataset_dir, split_name, save_sub_dir, tmp_dir,
                                    tf_record_pair_num=np.inf)