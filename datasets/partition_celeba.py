import sys
import os
import argparse
from glob import glob
import random
import shutil, pdb
import numpy as np
random.seed(0)

data_dir = '/esat/dragon/liqianma/datasets/Adaptation/Celeba/'
A_dir = os.path.join(data_dir, 'male_img_crop_resize')
trainA_dir = os.path.join(data_dir, 'male_img_crop_resize_train')
testA_dir = os.path.join(data_dir, 'male_img_crop_resize_test')
B_dir = os.path.join(data_dir, 'female_img_crop_resize')
trainB_dir = os.path.join(data_dir, 'female_img_crop_resize_train')
testB_dir = os.path.join(data_dir, 'female_img_crop_resize_test')




