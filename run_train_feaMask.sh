#!/bin/bash
source ~/.bashrc_liqianma
data_parent_dir='./data'
checkpoint_dir='./logs/'
gpu=0
segment_class=8
model=0

########################### Chose exp by uncomment ###########################
data_name='mnist_BW'
dataset_dir='mnist_BW_train_28x28'
bs=8
lr=1e-5
n_1=1
img_h=28
img_w=28
img_h_original=28
img_w_original=28
G_update=1

# data_name='mnist_multi_jitterColor_BW'
# dataset_dir='mnist_multi_jitterColor_BW_train_112x112'
# bs=8
# lr=1e-5
# n_1=3
# img_h=112
# img_w=112
# img_h_original=112
# img_w_original=112
# G_update=1

# data_name='celebaMaleFemaleCropTrTs'
# dataset_dir='celebaMaleFemale_train_128x128'
# bs=8
# lr=1e-4
# n_1=3
# img_h=128
# img_w=128
# img_h_original=128
# img_w_original=128
# G_update=1

# data_name='gta25k8'
# dataset_dir='gta25k_city_train_512x1024_8catId'
# bs=3
# lr=1e-4
# n_1=3
# img_h=256
# img_w=512
# img_h_original=512
# img_w_original=1024
# G_update=1

# data_name='gta25k8bdd'
# dataset_dir='gta25k_bdd_train_512x1024_8catId'
# bs=3
# lr=1e-4
# n_1=3
# img_h=256
# img_w=512
# img_h_original=512
# img_w_original=1024
# G_update=1
###############################################################


model_dir=${checkpoint_dir}'MODEL'${model}'_FeaMask_'${data_name}'_bs'${bs}'_lr'${lr}
python main.py --model_dir=${model_dir} --phase='train' \
               --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
               --gpu=${gpu}  --batch_size=${bs}  --model=${model} \
               --lr=${lr}  --use_lsgan=True --norm=None \
               --epoch=1 --segment_class=${segment_class} \
               --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
               --n_encoder=${n_1}  --n_encoder=${n_1}   \
               --img_h=${img_h}  --img_w=${img_w}  --img_h_original=${img_h_original}  --img_w_original=${img_w_original} \
               --G_update=${G_update}