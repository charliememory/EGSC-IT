#!/bin/bash
source ~/.bashrc_qianrusun2

## Copy data to local tmp dir for fast access
data_parent_dir='/BS/sun_project2/work/mlq_project/WassersteinGAN/data'

# data_name='gta25k20'
# segment_class=20
# # dataset_dir='gta25k_city_train_256x512_20trainId'
# dataset_dir='gta25k_city_train_512x1024_20trainId'
# # dataset_dir='gta25k_city_train_256x512_20trainId_onlyLabeled2975'

data_name='gta25k8'
segment_class=8
dataset_dir='gta25k_city_train_512x1024_8catId'

data_name='gta25k8bdd'
segment_class=8
# dataset_dir='gta25k_city_train_256x512_8catId'
dataset_dir='gta25k_bdd_train_512x1024_8catId'
# dataset_dir='gta25k_city_train_256x512_8catId_onlyLabeled2975'

# data_name='gta25k8bdd_ssim_select_style'
# segment_class=8
# dataset_dir='gta25k_bdd_ssim_select_style_train4test_256x512_8catId'

# data_name='gta25k8bdd_day'
# segment_class=8
# dataset_dir='gta25k_bdd_day_train_512x1024_8catId'

# data_name='gta25k8bdd_night'
# segment_class=8
# dataset_dir='gta25k_bdd_night_train_512x1024_8catId'

# data_name='gta'
# segment_class=8
# dataset_dir='gta_city_train_256x512'

# data_name='synsf'
# dataset_dir='synsf_city_train_256x832_segclass'  # model=2
# model=1000

# data_name='celebaMaleFemaleCropTrTs'
# segment_class=8
# dataset_dir='celebaMaleFemale_train_128x128'

# # data_name='mnist'
# # segment_class=8
# # dataset_dir='mnist_train_28x28'

# data_name='mnist_multi'
# segment_class=8
# dataset_dir='mnist_multi_train_112x112'

# data_name='mnist_multi_fixColor'
# segment_class=8
# dataset_dir='mnist_multi_fixColor_train_112x112'

# data_name='mnist_multi_jitterColor'
# segment_class=8
# dataset_dir='mnist_multi_jitterColor_train_112x112'

# data_name='mnist_BW'
# segment_class=8
# dataset_dir='mnist_BW_train_28x28'

# data_name='mnist_multi_jitterColor_BW'
# segment_class=8
# dataset_dir='mnist_multi_jitterColor_BW_train_112x112'

# data_name='mnist_svhn'
# segment_class=8
# dataset_dir='mnist_svhn_train_32x32'

model=1099
gpu=0
## TF training
checkpoint_dir='./logs_rightGammaBeta/'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_100L1_lsgan_256x512_bs8_epoch100_sum_lr2e-4_noTanh'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_100L1_lsgan_256x512_bs1_epoch100_sum_lr2e-4_noTanh_3nDis'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_100L1_binarygan_256x512_bs8_epoch100_sum_lr1e-4_DisSigmoid'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_lsgan_256x512_bs8_epoch100_lr2e-5'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_lsgan_128x384_bs8_normalInit_epoch100'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_4resblock_lsgan_128x384_bs8'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_3resblock_binarygan_128x384_bs8_lr1e-4_epoch100'
# model_dir=${checkpoint_dir}'MODEL1001_'${data_name}'_unitDiff_3resblock_4disLayer_binarygan_128x384_bs8_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1002_'${data_name}'_unit_3resblock_PatchDis_lsgan_128x384_bs8_epoch100'
# model_dir=${checkpoint_dir}'MODEL1003_'${data_name}'_3resblock_4disLayer_1disFeaLoss_binarygan_128x384_bs8_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1003_'${data_name}'_4resblock_4+2disLayer_1disFeaLoss_binarygan_128x384_bs8_lr1e-4'

model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_memoryDis_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_drop_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1001_'${data_name}'_unitDiff_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1003_'${data_name}'_unit_noNorm_3resblock_100L1_1disFeaLoss_lsgan_256x512_bs4_epoch100_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1004_'${data_name}'_unitAE_3resblock_100L1_lsgan_256x512_bs4_epoch100_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1007_'${data_name}'_unitMuSigma_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1008_'${data_name}'_unitDiff_1disFeaLoss_3resblock_100L1_lsgan_256x512_bs8_epoch100_lr2e-4'

model_dir=${checkpoint_dir}'MODEL1020_'${data_name}'_unitRecurrent_noNorm_3resblock_100L1_lsgan_256x512_bs1_epoch1100100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1030_'${data_name}'_unitTriDis_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1040_'${data_name}'_unitSpecBranch_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1041_'${data_name}'_unitMultiSpecBranchFromImg_noNorm_1INlayerBeforeRes_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1042_'${data_name}'_unitSpecBranchSgganImgMask_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1046_'${data_name}'_unitSpecBranchCycle_LAB_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1047_'${data_name}'_unitSpecBranchCycleSimple_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1043new_'${data_name}'_unitSpecBranchCycle_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1044_'${data_name}'_unitSpecBranchCycleSgganImgMask_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1045_'${data_name}'_unitMultiSpecBranchCycle_noNorm_3INlayerBeforeRes_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'

model_dir=${checkpoint_dir}'MODEL1050_'${data_name}'_unitAdaIN_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'_old_MODEL1051_'${data_name}'_unitAdaINCycle_noNorm_3INlayerBeforeRes_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1051_'${data_name}'_unitAdaINCycle_noNorm_1INlayerBeforeRes_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'

# model_dir=${checkpoint_dir}'MODEL1060_'${data_name}'_unitMultiSpecBranchFromImg_noNorm_3INlayerBeforeRes_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1060_'${data_name}'_unitMultiSpecBranchFromImg_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1061_'${data_name}'_unitMultiSpecBranchFromImgCycle_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1062_'${data_name}'_unitMultiSpecBranchFromImgCycle_simple_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1063_'${data_name}'_unitMultiSpecBranchFromImgCycle_FCN_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1064_'${data_name}'_unitMultiSpecBranchFromImgCycle_1Smooth_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1065_'${data_name}'_unitMultiSpecBranchFromImgCycle_100GauSmooth_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1066_'${data_name}'_unitMultiSpecBranchFromImgCycle_simple_1tvSmooth_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1066_'${data_name}'_unitMultiSpecBranchFromImgCycle_simple_10tvSmooth_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1067_'${data_name}'_unitMultiSpecBranchFromImg_simple_noNorm_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'

model_dir=${checkpoint_dir}'MODEL1070_'${data_name}'_unitVggSpecBranchFromImgCycle_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1071_'${data_name}'_unitVggSpecBranchFromImgCycle_0.0001VggLoss_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1072_'${data_name}'_unitVggSpecBranchFromImgCycle_1FeaLossDis_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1072_'${data_name}'_unitVggSpecBranchFromImgCycle_10FeaLossDis_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1074_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1074_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_512x1024_bs1_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1074_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_512x1024_bs2_epoch100_sum_lr2e-4_noTanh_dim64'
model_dir=${checkpoint_dir}'MODEL1074_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1074_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_DlossBigger3_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1075_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_LAB_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_512x1024_bs2_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1076_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_meanMultiDis_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1076_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_0.5meanMultiDis_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1076_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_meanMultiDis_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_512x1024_bs2_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1077_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_VggDis_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1077_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_0.5VggDis_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1078_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_0.0005VggLoss_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1079_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_1VggLoss_noGAN_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1073_'${data_name}'_unitMultiVggSpecBranchFromImgCycle_changeRes_1VggLoss_noGAN_LAB_lrDecay_3INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'

# model_dir=${checkpoint_dir}'MODEL1080_'${data_name}'_unitMultiSpecificBranchFromImgCycleSimpleRecon_10FeaLossDis_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1081_'${data_name}'_unitMultiSpecificBranchFromImgCycleSimpleFixDec_10FeaLossDis_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64'

model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_256x512_bs8_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_1Conv_256x512_bs8_lr2e-4'
model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_256x512_bs8_lr1e-5_5G1D'
model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_1Conv_256x512_bs8_lr1e-5_5G1D'
# model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_256x512_bs8_lr1e-5_noCrop'
# model_dir=${checkpoint_dir}'MODEL1090_'${data_name}'_unitVAEGANRecon_1Conv_1Res_256x512_bs8_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1091_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1092_'${data_name}'_unitMultiDecSpecificBranchFromImgCycle_changeRes_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_3INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_3INlayer_1GANLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask0to1_3INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_1INlayer_InsNormDis_100L1_lsgan_256x512_bs8_lr2e-4_1epoch'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_6SharedL_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_4SharedL_4EncGenRes_1INlayer_InsNormDis_100L1_lsgan_256x512_bs8_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4_2epoch'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4_6Ldis'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_3INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_1Conv_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_1Conv_1Res_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL1094_'${data_name}'_unitMultiDecSpecificBranchFromImgCycle_changeRes_feaMask_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1095_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_3EncStyle0.5ContentLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1095_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_3EncStyle0.5ContentLoss_1INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1095_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_0EncStyle1ContentLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1096_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1EncStyleContentLoss_noGAN_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1096_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1EncStyle0ContentLoss_noGAN_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1VggStyle1ContentLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1VggStyle0ContentLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'

# # model_dir=${checkpoint_dir}'MODEL1098_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1EncStyleContentLoss_1INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1098_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1EncStyleContentLoss_3INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1098_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1EncStyleContentLoss_3INlayer_1GANLoss_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# ## Run on Pascal
# # model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs2_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs2_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LSquareWeight_1INlayer_256x512_bs2_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL1100_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_Pair_1INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1101_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_Triplet_1INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1102_'${data_name}'_unitMultiVggSpecificBranchFromImgCycle_changeRes_0.875feaMask_10GAN_1e7VggStyle5L_1e5ContentLoss5L_256x512_bs2_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL1103_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMaskFromOther_1EncStyle0.2ContentLoss_1INlayer_256x512_bs4'
# # model_dir=${checkpoint_dir}'MODEL1104_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_0EncStyle0ContentLoss_1halfFCN8Loss_1INlayer_256x512_bs4'
# # model_dir=${checkpoint_dir}'MODEL1105_'${data_name}'_unitVAEGANReconCombData_3resblock_100L1_lsgan_256x512_bs4_lr2e-4_dim64'
# model_dir=${checkpoint_dir}'MODEL1106_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_ShareSB_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1107_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_noReluConvFeaMask_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1107_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_noReluConvFeaMask_1INlayer_256x512_bs4_lr2e-4_5G1D'
# model_dir=${checkpoint_dir}'MODEL1107_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_noReluConvKer3FeaMask_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1107_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_ConvKer3FeaMask_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1108_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_3-3BlurFeaMask_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1109_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_UnpoolShared3L_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1110_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_UnpoolEncGen_3SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1111_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_GF_3SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1112_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_ConvGF_AdaBN_3SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1112_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_ConvGF_AdaIN_3SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1113_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_fixCommon_9ImgConvDynGF_4SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1114_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_fixCommon_9ImgConvDynGFOnlyCrossDomain_4SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1113_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_5ImgConvDynGF_4SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1114_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_FeaMask_5ImgConvDynGFOnlyCrossDomain_4SharedL_1INlayer_256x512_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1SharedL_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs4_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1SharedL_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs4_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1SharedL_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs8_lr1e-5_5G1D_noFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_noFlip_SBfromMnistMulti'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_noFlip_SBfromMnistMultiFixColor'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_noFlip_SBfromMnistMultiFixColorFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_noFlip_SBfromMnistMultiFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_Flip_SBfromMnistMultiFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_Flip_SBfromMnistMultiFixColorFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs4_lr1e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_2G1D_epoch6'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_1G1D_epoch6'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e3L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch6'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1Conv_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e3L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_2e3L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_5e3L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_5G1D_epoch1'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_5e3L1VggStyle5L_1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_5G1D_epoch1'
model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-4_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch6_ZeroPad'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_changeRes_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_changeRes_feaMask_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1093_'${data_name}'_changeRes_feaMask_1Conv_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1502_'${data_name}'_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1502_'${data_name}'_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1503_'${data_name}'_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1504_'${data_name}'_changeRes_feaMaskE2E_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1504_'${data_name}'_changeRes_feaMaskE2Epretrian_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1503_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch6'
# model_dir=${checkpoint_dir}'MODEL1503_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_0e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-4_1G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1503_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1503_'${data_name}'_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1500_'${data_name}'_CycleGAN_changeRes_feaMask_1INlayer_bs8_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr2e-4_1G1D_epoch6'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_changeRes_feaMask_1e4L1VggStyle5L_2e2L1ContentLoss5LLinearWeight_1INlayer_bs3_lr1e-5_5G1D_epoch1'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'RandDomainA_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_Flip_SBfromMnistMultiFlip'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'RandDomainA_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1099_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_1Conv_1Res_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'RandDomainA_unitMultiEncSpecificBranchFromImgCycle_changeRes_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1Conv_1Res_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1Conv_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1097_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_1SharedL_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_256x512_bs4_lr1e-4'

# # model_dir=${checkpoint_dir}'MODEL1300_'${data_name}'_unitMultiEncSpecificBranchFromImgCycle_changeRes_feaMask_500Grad_1INlayer_3resblock_100L1_lsgan_256x512_bs4_epoch6_sum_lr2e-4_noTanh_dim64'

# model_dir=${checkpoint_dir}'MODEL1400_'${data_name}'_unitVAEGANLatentTrans_3resblock_100L1_lsgan_256x512_bs4_lr2e-4_dim64'

# model_dir=${checkpoint_dir}'MODEL4000_'${data_name}'_DITVggGmStyle_10GAN_VggStyle5L_ContentLoss5L_256x512_bs4'
# model_dir=${checkpoint_dir}'MODEL4001_'${data_name}'_DITEncInStyle_10GAN_1EncStyle_0.2ContentLoss_256x512_bs4'
# # model_dir=${checkpoint_dir}'MODEL4002_'${data_name}'_DITAEGANrecon_256x512_bs4_lr2e-5'
# model_dir=${checkpoint_dir}'MODEL4002_'${data_name}'_DITAEGANrecon_newArch_128dim_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4002_'${data_name}'_DITAEGANrecon_128dim_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4003_'${data_name}'_DITVAEGANrecon_newArch_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4003_'${data_name}'_DITVAEGANrecon_128x256_bs4_lr2e-5'
# model_dir=${checkpoint_dir}'MODEL4004_'${data_name}'_DITEncInStyleAEGANrecon_newArch_10GAN_LoadEnc_1EncStyle_1ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4004_'${data_name}'_DITEncInStyleAEGANrecon_newArch_10GAN_100EncStyle_100ContentLoss_128x256_bs4_lr2e-5'
# # # model_dir=${checkpoint_dir}'MODEL4005_'${data_name}'_DITEncInStyleVAEGANrecon_10GAN_1EncStyle_0.2ContentLoss_256x512_bs4'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_128dim_RandEnc_10GAN_100EncStyle_100ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_128dim_RandEnc_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_512dim_RandEnc_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_128dim_noAaBbRecon_RandEnc_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_oldArch_Cycle_128dim_RandEnc_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4006_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_128dim_LoadEnc_10GAN_1EncStyle_1ContentLoss_128x256_bs4_lr2e-5'
# # # model_dir=${checkpoint_dir}'MODEL4007_'${data_name}'_DITEncInStyleAEGANrecon_Cycle_FixAaBbRecon_128dim_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# # model_dir=${checkpoint_dir}'MODEL4008_'${data_name}'_DITEncInStyleAEGANrecon_newArch_Cycle_GenSeg_128dim_RandEnc_10GAN_0EncStyle_0ContentLoss_128x256_bs4_lr2e-5'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_10GAN_1e7VggStyle5L_1e2ContentLoss5L_128x256_bs4'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_10GAN_1e7VggStyle5L_1e2ContentLoss3L_128x256_bs4'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggInStyleAEGANrecon_newArch_10GAN_3e2VggStyle5L_1e5ContentLoss3L_128x256_bs4'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_128dim_10GAN_5e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_128x256_bs4'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_128dim_10GAN_5e3L1VggStyle5L_1e2L1ContentLoss5LSquareWeight_128x256_bs4'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_512dim_10GAN_1e4L1VggStyle5L_1e3L1ContentLoss5LSquareWeight_256x512_bs2'
# model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_newArch_512dim_10GAN_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_256x512_bs2'
# model_dir=${checkpoint_dir}'MODEL4012_'${data_name}'_DITVggGmStyleAEGANrecon_noSeg_newArch_512dim_10GAN_1e4L1VggStyle5L_1e3L1ContentLoss5LSquareWeight_256x512_bs2'
# model_dir=${checkpoint_dir}'MODEL4012_'${data_name}'_DITVggGmStyleAEGANrecon_noSeg_newArch_512dim_10GAN_1e3L1VggStyle5L_5e2L1ContentLoss5LLinearWeight_256x512_bs4'
# # model_dir=${checkpoint_dir}'MODEL4011_'${data_name}'_DITVggGmStyleAEGANreconBothVec_newArch_512dim_10GAN_1e7L2VggStyle5L_1e1L1ContentLoss5LSquareWeight_128x256_bs4'
# # model_dir=${checkpoint_dir}'MODEL4009_'${data_name}'_DITVggGmStyleAEGANrecon_noSeg_newArch_512dim_10GAN_1e7VggStyle5L_1e5ContentLoss5LLinearWeight_128x256_bs4'
# # model_dir=${checkpoint_dir}'MODEL4010_'${data_name}'_DITVggGmStyleAEGANreconUnetGen_newArch_512dim_10GAN_1e7VggStyle5L_1e5ContentLoss3L_128x256_bs4'

# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_InsNormDis_1Conv_1Res_100L1_lsgan_256x512_bs8_epoch1_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_epoch1_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs3_epoch1_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1000_'${data_name}'_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_lr1e-5_2G1D'
# model_dir=${checkpoint_dir}'MODEL1500_'${data_name}'_unit_noNorm_InsNormDis_CycleGAN_4ResNoShare_1Conv_100L1_bs8_epoch1_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL1501_'${data_name}'_unitMultiEncSpecificBranchFromImg_CycleGAN_changeRes_feaMask_3ResNoShare_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_128x256_bs4_lr2e-4'


# # # model_dir=${checkpoint_dir}'MODEL1005_'${data_name}'_unitSgganDisMaskDiff_3resblock_100L1_lsgan_256x512_bs8_epoch100_lr2e-4_noTanh_dim32'
# # # model_dir=${checkpoint_dir}'MODEL1006_'${data_name}'_unitSgganFeaEncDiff_3resblock_100L1_lsgan_256x512_bs8_epoch100_lr2e-4_noTanh_dim32'
# model_dir=${checkpoint_dir}'MODEL1009_'${data_name}'_unitSgganFeaEncSegBranch_noNorm_3resblock_100L1_lsgan_256x512_bs1_epoch100_lr2e-4_noTanh_dim32'
# # model_dir=${checkpoint_dir}'MODEL1010_'${data_name}'_unitSgganImgEncDiff_3resblock_100L1_lsgan_256x512_bs6_epoch100_lr2e-4_noTanh_dim32'
# # model_dir=${checkpoint_dir}'MODEL1010_'${data_name}'_unitSgganImgEncDiff_3resblock_10L1_lsgan_256x512_bs1_epoch100_lr2e-4_noTanh_dim64'
# # # model_dir=${checkpoint_dir}'MODEL1010_'${data_name}'_unitSgganImgEncDiff_3resblock_drop_100L1_lsgan_256x512_bs1_epoch100_lr2e-4_noTanh_dim64'
# # # model_dir=${checkpoint_dir}'MODEL1010_'${data_name}'_unitSgganImgEncDiff_4enc_3resblock_drop_100L1_lsgan_256x512_bs1_epoch100_lr2e-4_noTanh_dim64'
# # model_dir=${checkpoint_dir}'MODEL1011_'${data_name}'_unitSgganImgEnc_BN_3resblock_drop_100L1_lsgan_256x512_bs4_epoch100_lr2e-4_noTanh_dim32'
# # model_dir=${checkpoint_dir}'MODEL1011_'${data_name}'_unitSgganImgEnc_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_lr2e-4_noTanh_dim32'
# # # model_dir=${checkpoint_dir}'MODEL1012_'${data_name}'_unitSgganImgEncBothDiff_3resblock_drop_100L1_lsgan_256x512_bs1_epoch100_lr2e-4_noTanh_dim64'

# # model_dir=${checkpoint_dir}'MODEL2000_'${data_name}'_segment_DispNet_reg0.1_256x512_bs8_flip_crop_bright_lr1e-4'
# # # model_dir=${checkpoint_dir}'MODEL2000_'${data_name}'_segment_DispNet_reg0.1_ENetClassWeight_256x512_bs8_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2000_'${data_name}'_segment_DispNet_reg0.1_MedianFeqClassWeight_256x512_bs8_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_reg0.1_MedianFeqClassWeight_256x512_bs8_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_256x512_bs8_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_256x512_bs32_flip_crop_bright_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_256x512_bs8_flip_crop_bright_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_19class_256x512_bs8_flip_crop_bright_lr2e-4'
# # # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_19class_ENetClassWeight_256x512_bs8_flip_crop_bright_lr2e-4'
# # model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_19class_MedianFeqClassWeight_256x512_bs8_flip_crop_bright_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL2001_'${data_name}'_segment_LinkNet_19classValidAcc_256x512_bs8_flip_crop_bright_lr2e-4'
# # # model_dir=${checkpoint_dir}'MODEL2002_'${data_name}'_segment_LinkNetTargetLabel_19class_256x512_bs8_flip_crop_bright_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL2002_'${data_name}'_segment_LinkNetTargetLabel_19classValidAcc_256x512_bs2_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2001tmp_'${data_name}'_segment_LinkNet_csEval_256x512_bs4_flip_crop_bright_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL2003_'${data_name}'_segment_FCN8_256x512_bs8_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2004_'${data_name}'_segment_FCN8TargetLabel_256x512_bs2_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2004_'${data_name}'_segment_FCN8TargetLabel_ENetClassWeight_256x512_bs2_flip_crop_bright_lr1e-4'
# # model_dir=${checkpoint_dir}'MODEL2005_'${data_name}'_segment_ICNet_512x1024_bs8_flip_crop_bright_lr1e-3'
# # model_dir=${checkpoint_dir}'MODEL2005_'${data_name}'_segment_ICNet_256x512_bs8_flip_crop_bright_lr1e-3'
# model_dir=${checkpoint_dir}'MODEL2006_'${data_name}'_segment_ICNetTargetLabel_256x512_bs8_flip_crop_bright_lr1e-3'
# # model_dir=${checkpoint_dir}'MODEL2006_'${data_name}'_segment_ICNetTargetLabel_MedianFeqClassWeight_256x512_bs8_flip_crop_bright_lr1e-3'
# model_dir=${checkpoint_dir}'MODEL2006_'${data_name}'_segment_ICNetTargetLabel_updateBN_ENetClassWeight_256x512_bs8_flip_crop_bright_lr1e-3'

# model_dir=${checkpoint_dir}'MODEL3000_'${data_name}'_unitSpecBranchCycle_LinkNet_19classValidAcc_256x512_bs4_flip_crop_bright_lr2e-4'
# model_dir=${checkpoint_dir}'MODEL3000_'${data_name}'_unitSpecBranchCycle_LinkNet_finetune_19classValidAcc_256x512_bs4_flip_crop_bright_lr2e-5'
# model_dir=${checkpoint_dir}'MODEL3000_'${data_name}'_unitSpecBranchCycle_LinkNet_scratch_19classValidAcc_256x512_bs4_flip_crop_bright_lr2e-4'



# pretrained_unit_path=${checkpoint_dir}'MODEL1043_gta25k8_unitSpecBranchCycle_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64/UNIT.model-58002'
pretrained_unit_path=${checkpoint_dir}'MODEL1043new_gta25k20_unitSpecBranchCycle_noNorm_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64/UNIT.model-26002'
pretrained_seg_path=${checkpoint_dir}'MODEL2001_gta25k20_segment_LinkNet_19classValidAcc_256x512_bs8_flip_crop_bright_lr2e-4/UNIT.model-160002'
pretrained_seg_path=${checkpoint_dir}'MODEL2002_gta25k20_segment_LinkNetTargetLabel_19classValidAcc_256x512_bs8_flip_crop_bright_lr2e-4/UNIT.model-296002'
pretrained_path='./weights/vgg_19.ckpt'
pretrained_vgg_path='./weights/vgg_19.ckpt'
pretrained_fcn_path='./logs/MODEL2003_gta25k8_segment_FCN8_256x512_bs8_flip_crop_bright_lr1e-4/UNIT.model-122002'
# pretrained_path=${checkpoint_dir}'MODEL1080_gta25k20_unitMultiSpecificBranchFromImgCycleSimpleRecon_10FeaLossDis_1INlayerBeforeResBothForePost_3resblock_100L1_lsgan_256x512_bs4_epoch100_sum_lr2e-4_noTanh_dim64/UNIT.model-22002'
pretrained_path=${checkpoint_dir}'MODEL1090_gta25k8bdd_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-16002'
# pretrained_path=${checkpoint_dir}'MODEL1090_gta25k20_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_epoch2_sum_lr2e-4_noTanh_dim64/backup_UNIT.model-40002'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemale_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs4_lr2e-4_dim64/backup_UNIT.model-632002'
# pretrained_path=${checkpoint_dir}'MODEL1090_celebaMaleFemaleCrop_unitVAEGANRecon_3resblock_100L1_lsgan_256x512_bs8_lr2e-4_dim64/backup_UNIT.model-112002'
# pretrained_path=${checkpoint_dir}'MODEL1105_gta25k8bdd_unitVAEGANReconCombData_3resblock_100L1_lsgan_256x512_bs4_lr2e-4_dim64/backup_UNIT.model-24002'
# pretrained_AaBb_path=${checkpoint_dir}'MODEL4002_gta25k8bdd_night_DITAEGANrecon_512dim_128x256_bs4_lr2e-5/backup_UNIT.model-124002'
# pretrained_AaBb_path=${checkpoint_dir}'MODEL4002_gta25k8bdd_night_DITAEGANrecon_newArch_128x256_bs4_lr2e-5/backup_UNIT.model-126002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_unitVAEGANRecon_1Conv_256x512_bs8_lr2e-4/backup_UNIT.model-40002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_BW_unitVAEGANRecon_1Conv_256x512_bs8_lr1e-5_5G1D/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_unitVAEGANRecon_256x512_bs8_lr1e-5_2G1D_noFlip/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_fixColor_unitVAEGANRecon_256x512_bs8_lr1e-5_2G1D_noFlip/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_unitVAEGANRecon_256x512_bs8_lr1e-5_2G1D/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_fixColor_unitVAEGANRecon_256x512_bs8_lr1e-5_2G1D/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_jitterColor_unitVAEGANRecon_256x512_bs8_lr1e-5_2G1D/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_multi_jitterColor_BW_unitVAEGANRecon_256x512_bs8_lr1e-5_5G1D/backup_UNIT.model-56002'
# pretrained_path=${checkpoint_dir}'MODEL1090_mnist_unitVAEGANRecon_1Conv_1Res_256x512_bs8_lr2e-4/backup_UNIT.model-40002'
# pretrained_common_path=${checkpoint_dir}'MODEL1093_gta25k8bdd_unitMultiEncSpecificBranchFromImgCycle_changeRes_0.75feaMask_4SharedL_1INlayer_InsNormDis_100L1_lsgan_256x512_bs4_lr2e-4/backup_UNIT.model-136002'
mkdir ${model_dir}

python main.py --model_dir=${model_dir} --phase='train' \
               --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
               --gpu=${gpu}  --batch_size=3  --model=${model} \
               --lr=1e-4  --use_lsgan=True \
               --epoch=1 --segment_class=${segment_class} \
               --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
               --norm=None \
               --n_enc_share=1  --n_gen_share=1  --n_enc_resblock=3  --n_gen_resblock=3   \
               --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024 \
               --G_update=5 \
               --pretrained_path=${pretrained_path} \
               --save_freq=4000
               # --content_loss_IN=True
               # --img_h=256  --img_w=512  --img_h_original=256  --img_w_original=512 \

# python main.py --model_dir=${model_dir} --phase='train' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=8  --model=${model} \
#                --lr=1e-5  --use_lsgan=True \
#                --epoch=1 --segment_class=${segment_class} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --n_enc_share=1  --n_gen_share=1  --n_enc_resblock=3  --n_gen_resblock=3   \
#                --img_h=28  --img_w=28  --img_h_original=28  --img_w_original=28 \
#                --n_encoder=1  --n_gen_decoder=1 \
#                --G_update=5 \
#                --pretrained_path=${pretrained_path} \

# python main.py --model_dir=${model_dir} --phase='train' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=8  --model=${model} \
#                --lr=1e-4  --use_lsgan=True \
#                --epoch=1 --segment_class=${segment_class} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --n_enc_share=1  --n_gen_share=1  --n_enc_resblock=3  --n_gen_resblock=3   \
#                --img_h=128  --img_w=128  --img_h_original=128  --img_w_original=128 \
#                --G_update=5 \
#                --pretrained_path=${pretrained_path} \
#                # --img_h=112  --img_w=112  --img_h_original=112  --img_w_original=112 \

# python main.py --model_dir=${model_dir} --phase='train' \
#                --data_parent_dir=${data_parent_dir}  --dataset_dir=${dataset_dir}  \
#                --gpu=${gpu}  --batch_size=8  --model=${model} \
#                --lr=1e-5  --use_lsgan=True \
#                --epoch=1 --segment_class=${segment_class} \
#                --L1_weight=100  --L1_cycle_weight=100 --n_dis=4  --ngf=64 --ndf=64 \
#                --norm=None \
#                --n_enc_share=1  --n_gen_share=1  --n_enc_resblock=3  --n_gen_resblock=3   \
#                --G_update=5 \
#                --pretrained_path=${pretrained_path} \
#                --img_h=112  --img_w=112  --img_h_original=112  --img_w_original=112 \
#                # --img_h=128  --img_w=128  --img_h_original=128  --img_w_original=128 \
# #                # --n_encoder=1  --n_gen_decoder=1 \
# #                # --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024 \
# #                # --RandInvDomainA=True \
# #                # --img_h=32  --img_w=32  --img_h_original=32  --img_w_original=32 \
# #                # --img_h=28  --img_w=28  --img_h_original=28  --img_w_original=28 \
# #                # --img_h=128  --img_w=256  --img_h_original=512  --img_w_original=1024 \
# #                # --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024 \
# #                # --pretrained_common_path=${pretrained_common_path} \
# #                # --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024 \
# #                # --continue_train=1  --global_step=16002 \
# #                # --pretrained_AaBb_path=${pretrained_AaBb_path} \
# #                # --pretrained_vgg_path=${pretrained_vgg_path} \
# #                # --GAN_weight=5 \
# #                # --img_h=256  --img_w=512  --img_h_original=512  --img_w_original=1024 \
# #                # --pretrained_fcn_path=${pretrained_fcn_path} \
# #                # --img_h=512  --img_w=1024  --img_h_original=512  --img_w_original=1024 \
# #                # --continue_train=1  --global_step=28002 \
# #                # --continue_train=1  --global_step=4002 \
# #                # --pretrained_seg_path=${pretrained_seg_path} \
# #                # --pretrained_unit_path=${pretrained_unit_path} \
# #                # --replay_memory=True \
# #                # --res_dropout=0.5  
# #                # --n_encoder=4  --n_gen_decoder=4 \
# #                # --color_aug=False \
# #                # --pretrained_path=${pretrained_path} \
# #                # --use_norm=True \
# #                 # >> ${model_dir}'/logs.txt' 2>&1 
# #           #      --continue_train=1 \