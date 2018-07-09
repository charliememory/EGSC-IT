import argparse
import os
from trainer import *
from trainer_UNIT import *
from trainer_DIT import *
from trainer_seg import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', dest='gpu', type=int, default=-1, help='which gpu to use (-1 means select automatically)')
parser.add_argument('--data_parent_dir', dest='data_parent_dir', default='./datasets', help='path of the data parent dir)')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='gta', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=25, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--img_h', dest='img_h', type=int, default=256, help='image height')
parser.add_argument('--img_w', dest='img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h_original', dest='img_h_original', type=int, default=256, help='image height load from tf data')
parser.add_argument('--img_w_original', dest='img_w_original', type=int, default=512, help='image width load from tf data')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='The exponential decay rate for the 1st moment estimates')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=8000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=500, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--model_dir', dest='model_dir', type=str, default='./checkpoint', help='models are saved here')
parser.add_argument('--pretrained_path', dest='pretrained_path', type=str, default=None, help='pretrained model')
parser.add_argument('--pretrained_unit_path', dest='pretrained_unit_path', type=str, default=None, help='pretrained unit adaptation model')
parser.add_argument('--pretrained_seg_path', dest='pretrained_seg_path', type=str, default=None, help='pretrained segment model')
parser.add_argument('--pretrained_vgg_path', dest='pretrained_vgg_path', type=str, default=None, help='pretrained vgg model')
parser.add_argument('--pretrained_fcn_path', dest='pretrained_fcn_path', type=str, default=None, help='pretrained fcn model')
parser.add_argument('--pretrained_AaBb_path', dest='pretrained_AaBb_path', type=str, default=None, help='pretrained Aa Bb reconstruction model')
parser.add_argument('--pretrained_common_path', dest='pretrained_common_path', type=str, default=None, help='pretrained_common_path')
parser.add_argument('--test_model_path', dest='test_model_path', type=str, default=None, help='test_model_path')
parser.add_argument('--color_aug', dest='color_aug', type=str2bool, default=False, help='use color jitter in data augmentation')
# parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
# parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--Lg_lambda', dest='Lg_lambda', type=float, default=5.0, help='weight on gradloss term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=str2bool, default=False, help='generation network using residule block')
parser.add_argument('--use_norm', dest='use_norm', type=str2bool, default=False, help='generation network using instance norm')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=str2bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--segment_class', dest='segment_class', type=int, default=8, help='number of segmentation classes')
parser.add_argument('--model', dest='model', type=int, default=0, help='model idx')
parser.add_argument('--global_step', dest='global_step', type=int, default=0, help='model training step')
parser.add_argument('--depth_max', dest='depth_max', type=float, default=655.35, help='the inf value of depth')

#################### Params for UNIT ####################
parser.add_argument('--GAN_weight', type=float, default=10.0, help='Weight about GAN, lambda0')
parser.add_argument('--KL_weight', type=float, default=0.1, help='Weight about VAE, lambda1')
parser.add_argument('--L1_weight', type=float, default=100.0, help='Weight about VAE, lambda2' )
parser.add_argument('--KL_cycle_weight', type=float, default=0.1, help='Weight about VAE Cycle, lambda3')
parser.add_argument('--L1_cycle_weight', type=float, default=100.0, help='Weight about VAE Cycle, lambda4')

parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
parser.add_argument('--n_encoder', type=int, default=3, help='The number of encoder')
parser.add_argument('--n_enc_resblock', type=int, default=3, help='The number of encoder_resblock')
parser.add_argument('--n_enc_share', type=int, default=1, help='The number of share_encoder')
parser.add_argument('--n_gen_share', type=int, default=1, help='The number of share_generator')
parser.add_argument('--n_gen_resblock', type=int, default=3, help='The number of generator_resblock')
parser.add_argument('--n_gen_decoder', type=int, default=3, help='The number of generator_decoder')
parser.add_argument('--n_dis', type=int, default=4, help='The number of discriminator layer')

parser.add_argument('--res_dropout', type=float, default=0.0, help='The dropout ration of Resblock')
parser.add_argument('--smoothing', type=str2bool, default=False, help='smoothing loss use or not')
parser.add_argument('--norm', type=str, default='instance', help='The norm type')
parser.add_argument('--replay_memory', type=str2bool, default=False, help='discriminator pool use or not')
parser.add_argument('--pool_size', type=int, default=50, help='The size of image buffer that stores previously generated images')
# parser.add_argument('--lsgan', type=str2bool, default=False, help='lsgan loss use or not')
# parser.add_argument('--img_size', type=int, default=256, help='The size of image')
parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
# parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')
parser.add_argument('--normal_weight_init', type=str2bool, default=True, help='normal initialization use or not')
parser.add_argument('--RandInvDomainA', type=str2bool, default=False, help='Invert domain A image randomly')
parser.add_argument('--G_update', type=int, default=1, help='The number of G_optim in each iter')
parser.add_argument('--save_test_dis_score', type=str2bool, default=False, help='Whether save discriminator score during testing')
parser.add_argument('--num_style', type=int, default=1, help='Whether generate img for multiple styles during testing')
parser.add_argument('--content_loss_IN', type=str2bool, default=False, help='Whether apply instance norm when compute content loss')

args = parser.parse_args()
args.sample_dir = os.path.join(args.model_dir, 'sample')
args.test_dir = os.path.join(args.model_dir, 'test')

if args.norm=='None':
    args.norm = None

## Set gpu and import tf
if args.gpu>-1:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
import tensorflow as tf
import utils ## should be after tf import, since tf is used in utils
tf.set_random_seed(19)
tf.logging.set_verbosity(tf.logging.DEBUG)

## main func for tf.app.run()
def main(_): 
    if args.phase == 'train':
        gpu_options = tf.GPUOptions(allow_growth=False)
    else:
        gpu_options = tf.GPUOptions(allow_growth=True)
    tfconfig = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=tfconfig) as sess:
        if 0==args.model:
            trainer = sggan(sess, args)

        if args.phase == 'train':
            trainer.train(args)
        else:
            trainer.test(args)

if __name__ == '__main__':
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if 0==args.model:
        ## use original image input
        tf.app.run()
    else:
        ## use tf-record input
        if 1==args.model:
            trainer = sggan_tfrecord(args)
        elif 2==args.model:
            trainer = sggan_tfrecord_depth(args)
        elif 3==args.model:
            trainer = DispNet(args)
        elif 4==args.model:
            trainer = SegmentNet(args)
        elif 5==args.model:
            trainer = SegmentDispNet(args)  
        elif 6==args.model:
            trainer = sgganNoTargetDomainLabel_tfrecord(args)
        elif 7==args.model:
            trainer = cycada_tfrecord_depth(args)
        elif 8==args.model:
            trainer = sggan_cycada_tfrecord_depth(args)
        elif 9==args.model:
            trainer = cycada_imgsegG_tfrecord_depth(args)
        elif 10==args.model:
            trainer = cycada_imgsegGD_tfrecord_depth(args)
        elif 11==args.model:
            trainer = cycada_fT_tfrecord_depth(args)
            
        elif 1000==args.model:
            trainer = UNIT(args)
        # elif 1001==args.model:
        #     trainer = UNIT_diffMap(args)
        # elif 1002==args.model:
        #     trainer = UNIT_PatchDis(args)
        elif 1003==args.model:
            trainer = UNIT_FeaLossDis(args)
        # elif 1004==args.model:
        #     trainer = UNIT_AE(args)
        # elif 1005==args.model:
        #     trainer = UNIT_sggan_DisMask_diffMap(args)            
        # elif 1006==args.model:
        #     trainer = UNIT_sggan_feaEncMask(args)
        # elif 1007==args.model:
        #     trainer = UNIT_muSigma(args)    
        # elif 1008==args.model:
        #     trainer = UNIT_FeaLossDis_diffMap(args)   
        # elif 1009==args.model:
        #     trainer = UNIT_sggan_feaEncMaskSegBranch(args)   
        # elif 1010==args.model:
        #     trainer = UNIT_sggan_imgEncMask_diffMap(args)  
        # elif 1011==args.model:
        #     trainer = UNIT_sggan_imgEncMask(args)  
        # elif 1012==args.model:
        #     trainer = UNIT_sggan_imgEncMaskBothMnoM(args)  
        # elif 1020==args.model:
        #     trainer = UNIT_recurrent(args)  
        # elif 1030==args.model:
        #     trainer = UNIT_triDis(args)
        # elif 1040==args.model:
        #     trainer = UNIT_SpecificBranch(args)
        # elif 1042==args.model:
        #     trainer = UNIT_SpecificBranch_sggan_imgEncMask(args)
        # elif 1043==args.model:
        #     trainer = UNIT_SpecificBranchCycle(args)
        # elif 1044==args.model:
        #     trainer = UNIT_SpecificBranchCycle_sggan_imgEncMask(args)
        # elif 1045==args.model:
        #     trainer = UNIT_MultiSpecificBranch_Cycle(args)
        # elif 1046==args.model:
        #     trainer = UNIT_SpecificBranchCycle_LAB(args)
        # elif 1047==args.model:
        #     trainer = UNIT_SpecificBranchCycle_simple(args)


        elif 1050==args.model:
            trainer = UNIT_AdaIN(args)
        elif 1051==args.model:
            trainer = UNIT_AdaINCycle(args)

        elif 1060==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg(args)
        elif 1061==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle(args)
        elif 1062==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple(args)
        elif 1063==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle_FCN(args)
        elif 1064==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple_GradSmooth(args)
        elif 1065==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple_GaussianSmooth(args)
        elif 1066==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple_tvSmooth(args)
        elif 1067==args.model:
            trainer = UNIT_MultiSpecificBranchFromImg_simple(args)

        elif 1070==args.model:
            trainer = UNIT_VggSpecificBranchFromImg_Cycle(args)
        elif 1071==args.model:
            trainer = UNIT_VggSpecificBranchFromImg_Cycle_VggLoss(args)
        elif 1072==args.model:
            trainer = UNIT_VggSpecificBranchFromImg_Cycle_FeaLossDis(args)
        elif 1073==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss_noGAN_LAB(args)
        elif 1074==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes(args)
        elif 1075==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_LAB(args)
        elif 1076==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_MultiDis(args)
        elif 1077==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggDis(args)
        elif 1078==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss(args)
        elif 1079==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss_noGAN(args)


        # elif 1080==args.model:
        #     trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple_recon(args)
        # elif 1081==args.model:
        #     trainer = UNIT_MultiSpecificBranchFromImg_Cycle_simple_fixDec(args)

        elif 1090==args.model:
            trainer = UNIT_VAEGAN_recon(args)
        elif 1091==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes(args)
        elif 1092==args.model:
            trainer = UNIT_MultiDecSpecificBranchFromImg_Cycle_ChangeRes(args)
        elif 1093==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask(args)
        elif 1094==args.model:
            trainer = UNIT_MultiDecSpecificBranchFromImg_Cycle_ChangeRes_FeaMask(args)
        elif 1095==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss(args)
        elif 1096==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss_noGAN(args)
        elif 1097==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_VggStyleContentLoss(args)
        elif 1098==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss(args)            
        elif 1099==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(args)
        elif 1100==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Pair(args)            
        elif 1101==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Triplet(args)
        elif 1102==args.model:
            trainer = UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(args)
        elif 1103==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskFromOther_EncStyleContentLoss(args)
        elif 1104==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_FCN8Loss(args)
        elif 1105==args.model:
            trainer = UNIT_VAEGAN_recon_CombData(args)
        elif 1106==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ShareSB(args)
        elif 1107==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_ConvFeaMask(args)
        elif 1108==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_BlurFeaMask(args)
        elif 1109==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolShare(args)
        elif 1110==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolEncGen(args)
        elif 1111==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF(args)
        elif 1112==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ConvGF(args)
        elif 1113==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF(args)
        elif 1114==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF_OnlyCrossDomain(args)
        elif 1115==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_GFhigh(args)


        elif 1200==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_LearnIN(args)    


        elif 1300==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_Grad(args)  

        elif 1400==args.model:
            trainer = UNIT_VAEGAN_LatentTransform(args)  

        elif 1500==args.model:
            trainer = UNIT_CycleGAN(args) 
        elif 1501==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_CycleGAN_ChangeRes_FeaMask_VggStyleContentLoss(args) 
        elif 1502==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_FeaMask_VggStyleContentLoss(args) 
        elif 1503==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_EncIN(args) 
        elif 1504==args.model:
            trainer = UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskE2E_VggStyleContentLoss(args)


        # elif 2000==args.model:
        #     trainer = SegmentNet_DispNet(args)
        elif 2001==args.model:
            trainer = SegmentNet_LinkNet(args)
        elif 2002==args.model:
            trainer = SegmentNet_LinkNet_targetLabel(args)
        elif 2003==args.model:
            trainer = SegmentNet_FCN8(args)
        elif 2004==args.model:
            trainer = SegmentNet_FCN8_targetLabel(args)
        elif 2005==args.model:
            trainer = SegmentNet_ICNet(args)
        elif 2006==args.model:
            trainer = SegmentNet_ICNet_targetLabel(args)

            
        elif 3000==args.model:
            trainer = UNIT_SpecificBranchCycle_LinkNet(args)
            
            
        elif 4000==args.model:
            trainer = DIT_VggStyle(args)
        elif 4001==args.model:
            trainer = DIT_EncStyle(args)
        elif 4002==args.model:
            trainer = DIT_AEGAN_Recon(args)
        elif 4003==args.model:
            trainer = DIT_VAEGAN_Recon(args)
        elif 4004==args.model:
            trainer = DIT_EncStyle_AERecon(args)
        elif 4005==args.model:
            trainer = DIT_EncStyle_VAERecon(args)
        elif 4006==args.model:
            trainer = DIT_EncStyle_AERecon_Cycle(args)
        elif 4007==args.model:
            trainer = DIT_EncStyle_AERecon_Cycle_FixAaBb(args)
        elif 4008==args.model:
            trainer = DIT_EncStyle_AERecon_Cycle_GenSeg(args)
        elif 4009==args.model:
            trainer = DIT_VggStyle_AERecon(args)
        elif 4010==args.model:
            trainer = DIT_VggStyle_AERecon_UnetGen(args)
        elif 4011==args.model:
            trainer = DIT_VggStyle_AERecon_BothVec(args)
        elif 4012==args.model:
            trainer = DIT_VggStyle_AERecon_noSeg(args)



        # elif 1100==args.model:
        #     trainer = UNIT_wxb(args)   
            

        trainer.init_net(args)
            
        if args.phase == 'train':
            # trainer.train_xG1D(args)
            trainer.train(args)
        else:
            trainer.test(args)
