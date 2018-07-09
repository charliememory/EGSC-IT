#!/bin/bash
source ~/.bashrc_qianrusun2

## Copy data to local tmp dir for fast access
data_parent_dir='/BS/sun_project2/work/mlq_project/WassersteinGAN/data'
## TF training
checkpoint_dir='./logs_rightGammaBeta/'
gpu=2

################################################################################
## gta city test seq
model=1099
# dataset_dir='gta_city_test_seq_256x512'
# model_dir=${checkpoint_dir}'MODEL1099_gta25k8_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
# test_model_path=${model_dir}'/UNIT.model-12002'

## gta bdd test seq
model=1099
dataset_dir='gta_bdd_test_seq_256x512'
# dataset_dir='gta_bdd_test_seq_256x512_e5d3f007-8ca16edb'
# dataset_dir='gta_bdd_test_seq_256x512_d6346aa2-083f81fc'
model_dir=${checkpoint_dir}'MODEL1099_gta25k8bdd_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k8bdd_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-16002'
test_model_path=${model_dir}'/UNIT.model-16002'

python main.py --model_dir=${model_dir} --phase='test' \
               --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
               --gpu=${gpu}  --batch_size=4  --model=${model} \
               --lr=2e-4  --use_lsgan=True \
               --epoch=6 \
               --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
               --norm=None \
               --test_model_path=${test_model_path} \
               --pretrained_path=${pretrained_path} \
               --img_h=256  --img_w=512  --img_h_original=256  --img_w_original=512  \