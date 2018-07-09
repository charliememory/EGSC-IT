import sys, cv2
import os
import argparse
from glob import glob
import random
import shutil, pdb
import numpy as np
random.seed(0)

# data_dir = '/esat/dragon/liqianma/datasets/Adaptation/Celeba'
data_dir = sys.argv[1]

img_dir = os.path.join(data_dir, 'img_align_celeba')

tr_male_dir = os.path.join(data_dir, 'tr_male_img')
tr_female_dir = os.path.join(data_dir, 'tr_female_img')
val_male_dir = os.path.join(data_dir, 'val_male_img')
val_female_dir = os.path.join(data_dir, 'val_female_img')
ts_male_dir = os.path.join(data_dir, 'ts_male_img')
ts_female_dir = os.path.join(data_dir, 'ts_female_img')
attr_txt_path = os.path.join(data_dir, 'list_attr_celeba.txt')
partion_txt_path = os.path.join(data_dir, 'list_eval_partition.txt')
if not os.path.exists(tr_male_dir):
    os.makedirs(tr_male_dir)
if not os.path.exists(tr_female_dir):
    os.makedirs(tr_female_dir)
if not os.path.exists(val_male_dir):
    os.makedirs(val_male_dir)
if not os.path.exists(val_female_dir):
    os.makedirs(val_female_dir)
if not os.path.exists(ts_male_dir):
    os.makedirs(ts_male_dir)
if not os.path.exists(ts_female_dir):
    os.makedirs(ts_female_dir)

partion_dic = {}
with open(partion_txt_path, 'r') as f:
    img_name, gender = np.loadtxt(f, dtype=str, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)
    for i in range(len(img_name)):
        partion_dic[img_name[i]] = gender[i]


with open(attr_txt_path, 'r') as f:
    img_name, gender = np.loadtxt(f, dtype=str, delimiter=None, skiprows=2, usecols=(0,21), unpack=True)
    for i in range(len(img_name)):
        if '1'==gender[i]: ## Male
            if '0'==partion_dic[img_name[i]]:
                dst_path = os.path.join(tr_male_dir, img_name[i])
            elif '1'==partion_dic[img_name[i]]:
                dst_path = os.path.join(val_male_dir, img_name[i])
            elif '2'==partion_dic[img_name[i]]:
                dst_path = os.path.join(ts_male_dir, img_name[i])
            shutil.copy2(os.path.join(img_dir, img_name[i]), dst_path)
        elif '-1'==gender[i]: ## Female
            if '0'==partion_dic[img_name[i]]:
                dst_path = os.path.join(tr_female_dir, img_name[i])
            elif '1'==partion_dic[img_name[i]]:
                dst_path = os.path.join(val_female_dir, img_name[i])
            elif '2'==partion_dic[img_name[i]]:
                dst_path = os.path.join(ts_female_dir, img_name[i])
            shutil.copy2(os.path.join(img_dir, img_name[i]), dst_path)

        ## Crop and resize
        img=cv2.imread(dst_path)
        crop_img=img[20:218-20,:,:]
        resized_img=cv2.resize(crop_img,dsize=(132,132))
        cv2.imwrite(dst_path,resized_img)
