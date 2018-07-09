# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils
import pickle
import pdb

slim = tf.contrib.slim

_FILE_PATTERN = '%s_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': None, 'test': None}

_ITEMS_TO_DESCRIPTIONS = {
        'image_name_A': 'image name of A.',
        'image_name_B': 'image name of B.',
        'image_raw_A': 'image_raw_A.',
        'image_raw_B': 'image_raw_B.',
        'image_format': 'image_format jpg or png.',
        'image_height': 'image_height.',
        'image_width': 'image_width.',
}


from tensorflow.python.ops import parsing_ops
def get_split(split_name, data_dir, data_name='gta', img_height=256, img_width=512, seg_class=8, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading flowers.

    Args:
        split_name: A train/validation split name.
        data_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
        reader: The TensorFlow reader type.

    Returns:
        A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/validation split.
    """

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(data_dir, file_pattern % (data_name, split_name))

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
         'image_name_A' : tf.FixedLenFeature([], tf.string),
         'image_name_B' : tf.FixedLenFeature([], tf.string),
         'image_raw_A' : tf.FixedLenFeature([], tf.string),
         'image_raw_B' : tf.FixedLenFeature([], tf.string),

         'image_format': tf.FixedLenFeature([], tf.string, default_value='png'),
         'image_height': tf.FixedLenFeature([], tf.int64, default_value=img_height),
         'image_width': tf.FixedLenFeature([], tf.int64, default_value=img_width),
    }

    # if split_name in ['train']:
    #     keys_to_features = dict(keys_to_features, **{
    #      'image_raw_A_seg_class' : tf.FixedLenFeature([int(img_height*img_width*1)], tf.int64),
    #      'image_raw_B_seg_class' : tf.FixedLenFeature([int(img_height*img_width*1)], tf.int64),
    #     })
    #     pass
    # elif split_name in ['test', 'val']:
    #     pass

    items_to_handlers = {
        'image_name_A': slim.tfexample_decoder.Tensor('image_name_A'),
        'image_name_B': slim.tfexample_decoder.Tensor('image_name_B'),
        'image_raw_A': slim.tfexample_decoder.Image(image_key='image_raw_A', shape=[img_height, img_width, 3], channels=3, format_key='image_format'),
        'image_raw_B': slim.tfexample_decoder.Image(image_key='image_raw_B', shape=[img_height, img_width, 3], channels=3, format_key='image_format'),
    }

    # if split_name in ['train']:
    #     items_to_handlers = dict(items_to_handlers, **{
    #     'image_raw_A_seg_class': slim.tfexample_decoder.Tensor('image_raw_A_seg_class',shape=[int(img_height), int(img_width), 1]), 
    #     'image_raw_B_seg_class': slim.tfexample_decoder.Tensor('image_raw_B_seg_class',shape=[int(img_height), int(img_width), 1]), 
    #     })
    # elif split_name in ['test', 'val']:
    #     pass


    decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

    print('load tf_record_sample_num ......')
    fpath = os.path.join(data_dir, 'tf_record_sample_num.txt')
    with open(fpath,'r') as f:
        num_samples = int(f.read().split(':')[1])
    
    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

