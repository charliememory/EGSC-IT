import os, pdb, glob
import scipy.misc as misc
import numpy as np
from skimage.transform import resize

dataset_dir = '/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/'
src_dir = os.path.join(dataset_dir,'gta25k')
dst_dir = os.path.join(dataset_dir,'gta25k_256x512')
trainA_dir = os.path.join(dst_dir,'trainA')
trainB_dir = os.path.join(dst_dir,'trainB')
testA_dir = os.path.join(dst_dir,'testA')
testB_dir = os.path.join(dst_dir,'testB')
if not os.path.exists(trainA_dir):
    os.makedirs(trainA_dir)
if not os.path.exists(trainB_dir):
    os.makedirs(trainB_dir)
if not os.path.exists(testA_dir):
    os.makedirs(testA_dir)
if not os.path.exists(testB_dir):
    os.makedirs(testB_dir)


# pdb.set_trace()
#for path in glob.glob(os.path.join(src_dir,'trainA','*.png')):
#    img = misc.imread(path)
#    img = resize(img, (256, 512))
#    misc.imsave( os.path.join(trainA_dir, path.split('/')[-1]), img)

for path in glob.glob(os.path.join(src_dir,'trainB_extra','*.png')):
    img = misc.imread(path)
    img = resize(img, (256, 512))
    misc.imsave( os.path.join(trainB_dir, path.split('/')[-1]), img)

# for path in glob.glob(os.path.join(src_dir,'testA','*.png')):
#     img = misc.imread(path)
#     img = resize(img, (256, 512))
#     misc.imsave( os.path.join(testA_dir, path.split('/')[-1]), img)

# for path in glob.glob(os.path.join(src_dir,'testB','*.png')):
#     img = misc.imread(path)
#     img = resize(img, (256, 512))
#     misc.imsave( os.path.join(testB_dir, path.split('/')[-1]), img)
