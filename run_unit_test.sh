#!/bin/bash
source ~/.bashrc_qianrusun2

## Copy data to local tmp dir for fast access
data_parent_dir='/BS/sun_project2/work/mlq_project/WassersteinGAN/data'
## TF training
checkpoint_dir='./logs_rightGammaBeta/'

################################################################################
model=1099
gpu=1

# data_name='gta25k20'
# segment_class=20
# # dataset_dir='gta25k_city_train_256x512_20trainId'
# dataset_dir='gta25k_city_train_512x1024_20trainId'
# # dataset_dir='gta25k_city_train_256x512_20trainId_onlyLabeled2975'

data_name='gta25k8'
segment_class=8
dataset_dir='gta25k_city_train_512x1024_8catId'

# data_name='gta25k8'
# segment_class=8
# dataset_dir='gta25k_city_test_256x512_8catId'

data_name='gta25k8bdd'
segment_class=8
# dataset_dir='gta25k_city_train_256x512_8catId'
dataset_dir='gta25k_bdd_train_512x1024_8catId'
# dataset_dir='gta25k_city_train_256x512_8catId_onlyLabeled2975'

# # data_name='gta25k8bdd_day'
# segment_class=8
# dataset_dir='gta25k_bdd_day_train_512x1024_8catId'

# data_name='gta25k8bdd_night'
# segment_class=8
# dataset_dir='gta25k_bdd_night_train_512x1024_8catId'

# data_name='gta25k8bdd'
# segment_class=8
# dataset_dir='gta25k_bdd_rand_select_style_train4test_256x512_8catId'

# data_name='gta25k8bdd'
# segment_class=8
# dataset_dir='gta25k_bdd_ssim_select_style_train4test_256x512_8catId'

data_name='gta25k8bdd'
segment_class=8
# dataset_dir='gta25k_bdd_test_512x1024_8catId'
dataset_dir='gta25k_bdd_test_256x512_8catId'

####
#### SG-UNIT GTA2City
# model_dir=${checkpoint_dir}'MODEL1099_gta25k8_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs2_lr2e-4'
# pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
# test_model_path=${model_dir}'/UNIT.model-136002'
model_dir=${checkpoint_dir}'MODEL1099_gta25k8_changeRes_feaMask_1e3L1VggStyle5L_1e3L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch6'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
test_model_path=${model_dir}'/backup_UNIT.model-12002'
model_dir=${checkpoint_dir}'MODEL1099_gta25k8_changeRes_feaMask_1e3L1VggStyle5L_1e3L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
test_model_path=${model_dir}'/UNIT.model-12002'
model_dir=${checkpoint_dir}'MODEL1099_gta25k8_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
test_model_path=${model_dir}'/UNIT.model-12002'
#### SG-UNIT GTA2BDD
# model_dir=${checkpoint_dir}'MODEL1099_gta25k8bdd_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1SharedL_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs4_lr1e-4'
# pretrained_path=${checkpoint_dir}'MODEL1090_gta25k8bdd_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-16002'
# test_model_path=${model_dir}'/UNIT.model-136002'
model_dir=${checkpoint_dir}'MODEL1099_gta25k8bdd_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k8bdd_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-16002'
test_model_path=${model_dir}'/UNIT.model-16002'
#### UNIT 
# model_dir=${checkpoint_dir}'MODEL1000_gta25k8_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs3_epoch6_lr2e-4'
# test_model_path=${model_dir}'/UNIT.model-136002'
# model_dir=${checkpoint_dir}'MODEL1000_gta25k8bdd_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs3_epoch1_lr2e-4'
# test_model_path=${model_dir}'/UNIT.model-136002'
####

python main.py --model_dir=${model_dir} --phase='test' \
               --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
               --gpu=${gpu}  --batch_size=4  --model=${model} \
               --lr=2e-4  --use_lsgan=True \
               --epoch=6 --segment_class=${segment_class} \
               --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
               --norm=None \
               --test_model_path=${test_model_path} \
               --pretrained_path=${pretrained_path} \
               --img_h=256  --img_w=512  --img_h_original=256  --img_w_original=512  \
               # --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024  \


################################################################################
# model=1099
# gpu=2
# data_name='celebaMaleFemaleCropTrTs'
# segment_class=8
# dataset_dir='celebaMaleFemale_test_128x128'
# #### SG-UNIT CelebA
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCropTrTs_changeRes_feaMask_5e3L1VggStyle5L_0e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr2e-4_1G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-88002'
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCropTrTs_changeRes_feaMask_1e3L1VggStyle5L_0e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr2e-4_1G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-88002'
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCropTrTs_changeRes_feaMask_5e3L1VggStyle5L_0e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr2e-4_1G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-88002'
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCrop_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1SharedL_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_256x512_bs4_lr2e-4'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-80002'
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCropTrTs_changeRes_feaMask_5e3L1VggStyle5L_1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_5G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-32002'
# model_dir=${checkpoint_dir}'MODEL1099_celebaMaleFemaleCropTrTs_changeRes_feaMask_5e3L1VggStyle5L_1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_5G1D_epoch1'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# test_model_path=${model_dir}'/UNIT.model-32002'

# #### UNIT 
# # model_dir=${checkpoint_dir}'MODEL1000_celebaMaleFemaleCropTrTs_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_epoch1_lr2e-4'
# # test_model_path=${model_dir}'/UNIT.model-88002'
# ####

# python main.py --model_dir=${model_dir} --phase='test' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=1  --model=${model} \
#                --lr=2e-4  --use_lsgan=True \
#                --epoch=6 --segment_class=${segment_class} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --test_model_path=${test_model_path} \
#                --img_h=128  --img_w=128  --img_h_original=128  --img_w_original=128  \
#                --num_style=30 \
#                --pretrained_path=${pretrained_path} \


################################################################################
# model=1503
# gpu=0
# data_name='mnist_BW'
# dataset_dir='mnist_BW_test_28x28'
# # dataset_dir='mnist_BW_train4test_28x28'
# # data_name='emnist_BW'
# # dataset_dir='emnist_BW_mnist_letter_test_28x28'
# # dataset_dir='emnist_BW_letter_mnist_test_28x28'
# # dataset_dir='emnist_BW_letter_letter_test_28x28'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_BW_unitVAEGANRecon_1Conv_256x512_bs8_lr1e-5_5G1D/backup_UNIT.model-56002'
# # ####
# model_dir=${checkpoint_dir}'MODEL1099_mnist_BW_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ###
# model_dir=${checkpoint_dir}'MODEL1097_mnist_BW_changeRes_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ###
# model_dir=${checkpoint_dir}'MODEL1099_mnist_BW_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ###
# model_dir=${checkpoint_dir}'MODEL1502_mnist_BW_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ###
# model_dir=${checkpoint_dir}'MODEL1503_mnist_BW_changeRes_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'

# ###
# # model_dir=${checkpoint_dir}'MODEL1500_mnist_BW_CycleGAN_changeRes_feaMask_1Conv_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# # test_model_path=${model_dir}'/UNIT.model-56002'
# ####
# # model_dir=${checkpoint_dir}'MODEL1000_mnist_BW_unit_noNorm_InsNormDis_1Conv_100L1_lsgan_256x512_bs8_lr1e-5_5G1D'
# # test_model_path=${model_dir}'/UNIT.model-56002'
# ####

# python main.py --model_dir=${model_dir} --phase='test' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=4  --model=${model} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --test_model_path=${test_model_path} \
#                --n_encoder=1  --n_gen_decoder=1 \
#                --img_h=28  --img_w=28  --img_h_original=28  --img_w_original=28 \
#                --pretrained_path=${pretrained_path} \

################################################################################
# model=1502
# gpu=2
# data_name='mnist_multi_jitterColor_BW'
# dataset_dir='mnist_multi_jitterColor_BW_test_112x112'
# # data_name='emnist_multi_jitterColor_BW'
# # dataset_dir='emnist_multi_jitterColor_BW_mnist_letter_test_112x112'
# # dataset_dir='emnist_multi_jitterColor_BW_letter_mnist_test_112x112'
# # dataset_dir='emnist_multi_jitterColor_BW_letter_letter_test_112x112'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_jitterColor_BW_unitVAEGANRecon_256x512_bs8_lr1e-5_5G1D/backup_UNIT.model-56002'
# ####
# model_dir=${checkpoint_dir}'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ####
# model_dir=${checkpoint_dir}'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# ####
# model_dir=${checkpoint_dir}'MODEL1097_mnist_multi_jitterColor_BW_changeRes_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# # ####
# model_dir=${checkpoint_dir}'MODEL1502_mnist_multi_jitterColor_BW_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# test_model_path=${model_dir}'/UNIT.model-56002'
# # ####
# # model_dir=${checkpoint_dir}'MODEL1500_mnist_multi_jitterColor_BW_CycleGAN_changeRes_feaMask_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# # test_model_path=${model_dir}'/UNIT.model-56002'
# ####
# # model_dir=${checkpoint_dir}'MODEL1000_mnist_multi_jitterColor_BW_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_lr1e-5_5G1D'
# # test_model_path=${model_dir}'/UNIT.model-56002'


# python main.py --model_dir=${model_dir} --phase='test' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=4  --model=${model} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --test_model_path=${test_model_path} \
#                --img_h=112  --img_w=112  --img_h_original=112  --img_w_original=112 \
#                --pretrained_path=${pretrained_path} \