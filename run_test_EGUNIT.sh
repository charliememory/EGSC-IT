#!/bin/bash
source ~/.bashrc_liqianma
data_parent_dir='./data'
checkpoint_dir='./logs/'
gpu=0
segment_class=8
model=1

########################### Chose exp by uncomment ###########################
data_name='mnist_BW'
dataset_dir='mnist_BW_train_28x28'
bs=4
num_style=1
n_1=1
img_h=28
img_w=28
img_h_original=28
img_h_original=28
style_weight=1e3
content_weight=1e1

# data_name='mnist_multi_jitterColor_BW'
# dataset_dir='mnist_multi_jitterColor_BW_train_112x112'
# bs=4
# num_style=1
# n_1=3
# img_h=112
# img_w=112
# img_h_original=112
# img_h_original=112
# style_weight=1e4
# content_weight=1e2

# data_name='celebaMaleFemaleCropTrTs'
# dataset_dir='celebaMaleFemale_train_128x128'
# bs=1
# num_style=30
# n_1=3
# img_h=128
# img_w=128
# img_h_original=128
# img_h_original=128
# style_weight=5e3
# content_weight=1e1

# data_name='gta25k8'
# dataset_dir='gta25k_city_train_512x1024_8catId'
# bs=4
# num_style=1
# n_1=3
# img_h=256
# img_w=512
# img_h_original=512
# img_h_original=1024
# style_weight=1e3
# content_weight=5e2

# data_name='gta25k8bdd'
# dataset_dir='gta25k_bdd_train_512x1024_8catId'
# bs=4
# num_style=1
# n_1=3
# img_h=256
# img_w=512
# img_h_original=512
# img_h_original=1024
# style_weight=1e4
# content_weight=1e2


model_dir=${checkpoint_dir}'MODEL'${model}'_EGUNIT_'${data_name}'_bs'${bs}'_lr'${lr}'_'${style_weight}'Style_'${content_weight}'Content'
pretrained_path=${checkpoint_dir}'MODEL0_'${data_name}'_bs'${bs}'_lr'${lr}'/UNIT.model-0'
test_model_path=${model_dir}'/UNIT.model-0'
python main.py --model_dir=${model_dir} --phase='test' \
               --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
               --gpu=${gpu}  --batch_size=${bs}  --model=${model} \
               --norm=None \
               --epoch=1 --segment_class=${segment_class} \
               --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
               --n_enc_share=1  --n_gen_share=1  --n_enc_resblock=${n_1}  --n_gen_resblock=${n_1}   \
               --img_h=${img_h}  --img_w=${img_w}  --img_h_original=${img_h_original}  --img_w_original=${img_w_original} \
               --test_model_path=${test_model_path} \
               --pretrained_path=${pretrained_path} \
               --num_style=${num_style} \
