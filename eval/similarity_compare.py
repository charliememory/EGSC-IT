from __future__ import print_function

import os, pdb, sys, glob
import StringIO
import scipy.misc
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
# from PIL import Image
import scipy.misc

data_dir = '/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/gta25k_bdd'
res_dir = os.path.join(data_dir, 'similarity_compare')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
trainB_dir = os.path.join(data_dir, 'trainA')
trainA_dir = os.path.join(data_dir, 'trainB_extra')
types = ('*.jpg', '*.png') # the tuple of file types
trainA_files = []
trainB_files = []
for files in types:
    trainA_files.extend(glob.glob(os.path.join(trainA_dir, files)))
    trainB_files.extend(glob.glob(os.path.join(trainB_dir, files)))

# write html for visual comparison
index_path = os.path.join(res_dir, 'index.html')
index = open(index_path, 'w')
index.write("<html><body><table><tr>")
index.write("<th>GTA</th><th>BDD</th></tr>")
A_num = 100
B_num = 100
img_h = 128
img_w = 256
for i in range(A_num):
    img_A = scipy.misc.imread(trainA_files[i])
    img_A_resize = scipy.misc.imresize(img_A, [img_h, img_w])
    img_A_gray = rgb2gray(img_A_resize.clip(min=0,max=255))
    out_A_path = os.path.join(res_dir, trainA_files[i].split('/')[-1])
    scipy.misc.imsave(out_A_path, img_A_resize)
    score_list = []
    out_B_path_list = []
    for j in range(B_num):
        img_B = scipy.misc.imread(trainB_files[j])
        img_B_resize = scipy.misc.imresize(img_B, [img_h, img_w])
        img_B_gray = rgb2gray(img_B_resize.clip(min=0,max=255))
        score_list.append(ssim(img_A_gray, img_B_gray, data_range=img_B_gray.max()-img_B_gray.min(), multichannel=False))
        out_B_path = os.path.join(res_dir, trainB_files[j].split('/')[-1])
        scipy.misc.imsave(out_B_path, img_B_resize)
        out_B_path_list.append(out_B_path)

    combined = sorted(zip(score_list,out_B_path_list), key=lambda x: x[0], reverse=True)
    score_list[:], out_B_path_list[:] = zip(*combined)

    index.write("<tr>")
    index.write("<td><img src='%s' height='%d' width='%d'></td>" % (out_A_path, img_h, img_w))
    for j in range(B_num):
        index.write("<td><img src='%s' height='%d' width='%d'></td>" % (out_B_path_list[j], img_h, img_w))
    index.write("</tr>")

    index.write("<tr>")
    index.write("<td>similarity</td>")
    for j in range(B_num):
        index.write("<td>gray_ssim: %f</td>" % score_list[j])
    index.write("</tr>")

index.close()



