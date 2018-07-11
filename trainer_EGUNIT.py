from __future__ import division
import os, pdb, sys
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import scipy.misc
import logging
import math

from model import *
from utils import *
from loss_func import *

## Use tf record to speed up
from datasets import gta, dataset_utils, celeba

import ops_UNIT
from labels_utils import *

import tensorflow_vgg19
# from utils import *
# from glob import glob
# import time

class UNIT(object):
    def __init__(self, args):
        self.model_name = 'UNIT.model'
        self.model_dir = args.model_dir
        self.test_dir = args.test_dir
        self.model_dir = args.model_dir
        self.sample_dir = args.sample_dir
        self.data_parent_dir = args.data_parent_dir
        self.dataset_dir = args.dataset_dir
        self.segment_class = args.segment_class
        self.scale_num = 4
        self.logging = logging
        self.logging.basicConfig(filename=os.path.join(args.model_dir, 'INFO.log'),
                                level=logging.INFO)
        self.color_aug = args.color_aug
        self.depth_max = args.depth_max

        self.epoch = args.epoch # 100000
        self.batch_size = args.batch_size # 1
        self.img_w = args.img_w
        self.img_h = args.img_h
        self.img_w_original = args.img_w_original
        self.img_h_original = args.img_h_original
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.ngf = args.ngf
        self.ndf = args.ndf
        self.beta1 = args.beta1

        # self.lr = args.lr # 0.0001
        """ Weight about VAE """
        self.KL_weight = args.KL_weight # lambda 1
        self.L1_weight = args.L1_weight # lambda 2

        """ Weight about VAE Cycle"""
        self.KL_cycle_weight = args.KL_cycle_weight # lambda 3
        self.L1_cycle_weight = args.L1_cycle_weight # lambda 4

        """ Weight about GAN """
        self.GAN_weight = args.GAN_weight # lambda 0

        """ Weight about VGG loss """
        self.GAN_weight = args.GAN_weight # lambda 0
        self.style_weight = args.style_weight
        self.content_weight = args.content_weight


        """ Encoder """
        # self.input_c_dim = args.ch # base channel number per layer
        self.n_encoder = args.n_encoder
        self.n_enc_resblock = args.n_enc_resblock
        self.n_enc_share = args.n_enc_share

        """ Generator """
        self.n_gen_share = args.n_gen_share
        self.n_gen_resblock = args.n_gen_resblock
        self.n_gen_decoder = args.n_gen_decoder

        """ Discriminator """
        self.n_dis = args.n_dis # + 2

        self.res_dropout = args.res_dropout
        self.smoothing = args.smoothing
        self.use_lsgan = args.use_lsgan
        self.norm = args.norm
        self.replay_memory = args.replay_memory
        self.pool_size = args.pool_size
        # self.img_size = args.img_size
        # self.output_c_dim = args.img_ch
        # self.augment_flag = args.augment_flag
        # self.augment_size = self.img_size + (30 if self.img_size == 256 else 15)
        self.normal_weight_init = args.normal_weight_init

        self.pretrained_vgg_path = args.pretrained_vgg_path
        self.content_loss_IN = args.content_loss_IN

        # self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name, size=self.img_size)
        # self.num_batches = max(len(self.trainA) // self.batch_size, len(self.trainB) // self.batch_size)

        if args.phase == 'train':
            self.is_training = True
            self.num_threads = 4
            self.capacityCoff = 2
            # self.sample_num = 2000
            fpath = os.path.join(self.data_parent_dir, self.dataset_dir, 'tf_record_sample_num.txt')
            with open(fpath,'r') as f:
                self.sample_num = int(f.read().split(':')[1])
        else:
            self.is_training = False
            # during testing to keep the order of the input data
            # self.num_threads = 1
            # self.capacityCoff = 1
            self.num_threads = 4
            self.capacityCoff = 2
            fpath = os.path.join(self.data_parent_dir, self.dataset_dir, 'tf_record_sample_num.txt')
            with open(fpath,'r') as f:
                self.sample_num = int(f.read().split(':')[1])

        if 'gta' in self.dataset_dir.lower():
            self.dataset_tr_obj = gta.get_split(args.phase, os.path.join(self.data_parent_dir, self.dataset_dir), 
                                                'gta', self.img_h_original, self.img_w_original, self.segment_class)
        elif 'celeba' in self.dataset_dir.lower():
            self.dataset_tr_obj = celeba.get_split(args.phase, os.path.join(self.data_parent_dir, self.dataset_dir), 
                                                'celeba', self.img_h_original, self.img_w_original, self.segment_class)
        elif 'mnist' in self.dataset_dir.lower():
            self.dataset_tr_obj = celeba.get_split(args.phase, os.path.join(self.data_parent_dir, self.dataset_dir), 
                                                'mnist', self.img_h_original, self.img_w_original, self.segment_class)

            # self.dataset_ts_obj = gta.get_split('test', os.path.join(self.data_parent_dir, self.dataset_dir.replace('train','test')),
            #                                     'gta', self.img_h_original, self.img_w_original, self.segment_class)
        # elif 'synsf' in self.dataset_dir.lower():
        #     self.dataset_tr_obj = synsf.get_split('train', os.path.join(self.data_parent_dir, self.dataset_dir), 
        #                                         'synsf', self.img_h_original, self.img_w_original, self.segment_class)
        #     self.dataset_ts_obj = synsf.get_split('test', os.path.join(self.data_parent_dir, self.dataset_dir.replace('train','test')),
        #                                         'synsf', self.img_h_original, self.img_w_original, self.segment_class)
        else:
            raise Exception(dataset_dir.lower() + ' is not valid')

        # if 'gta' in self.dataset_dir.lower():
        #     ## Input training data
        #     self.img_name_A, self.img_name_B, self.real_data, self.seg_data, self.mask_A_ori, self.mask_B_ori, self.A_seg_valid, self.B_seg_valid = self._load_batch_data_tf(self.dataset_tr_obj, is_training=self.is_training)
        #     self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        #     self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        #     self.seg_A = self.seg_data[:, :, :, :self.input_c_dim]
        #     self.seg_B = self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        #     ## Input test data
        #     # self.img_name_A_ts, self.img_name_B_ts, self.real_data_ts, self.seg_data_ts, self.mask_A_ts_ori, self.mask_B_ts_ori = self._load_batch_data_tf(self.dataset_ts_obj, is_training=False) 
        #     # self.real_A_ts = self.real_data_ts[:, :, :, :self.input_c_dim]
        #     # self.real_B_ts = self.real_data_ts[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        # elif 'celeba' in self.dataset_dir.lower() or 'mnist' in self.dataset_dir.lower():
        #     ## Input training data
        #     self.img_name_A, self.img_name_B, self.real_data = self._load_batch_data_tf_noSeg(self.dataset_tr_obj, is_training=self.is_training)
        #     self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        #     self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        # else:
        #     raise Exception(dataset_dir.lower() + ' is not valid')

        self.img_name_A, self.img_name_B, self.real_data = self._load_batch_data_tf_noSeg(self.dataset_tr_obj, is_training=self.is_training)
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        if args.RandInvDomainA:
            ## Random muliply +-1
            shape = self.real_A.get_shape().as_list()
            sign = tf.where(tf.random_uniform([shape[0],1,1,1])>0.5, tf.ones([shape[0],1,1,1]), -tf.ones([shape[0],1,1,1]))
            sign = tf.tile(sign, [1,shape[1],shape[2],shape[3]])
            self.real_A = self.real_A * sign

        self.domain_A = self.real_A
        self.domain_B = self.real_B
        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=10)
        # self.pool = ImagePool(args.max_size)

    ##############################################################################
    # BEGIN of ENCODERS
    def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
        channel = self.ngf
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            return x
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    def share_encoder(self, x, is_training=True, reuse=False, scope="share_encoder"):
        channel = self.ngf * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_enc_share) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            x = ops_UNIT.gaussian_noise_layer(x)

            return x

    def share_generator(self, x, is_training=True, reuse=False, scope="share_generator"):
        channel = self.ngf * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_share) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

        return x
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')
            # x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='deconv_out')

            return x
    # END of DECODERS
    ##############################################################################

    # ##############################################################################
    # # BEGIN of DISCRIMINATORS
    # def discriminator(self, x, reuse=False, scope="discriminator"):
    #     channel = self.ndf
    #     with tf.variable_scope(scope, reuse=reuse):
    #         x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

    #         for i in range(1, self.n_dis) :
    #             x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
    #             channel *= 2

    #         x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')
    #         # x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='sigmoid', scope='dis_logit')

    #         return x
    # # END of DISCRIMINATORS
    # ##############################################################################

    ##############################################################################
    # BEGIN of DISCRIMINATORS
    def discriminator(self, x, reuse=False, is_training=True, scope="discriminator", activation_fn=ops_UNIT.LeakyReLU):
        channel = self.ndf
        with tf.variable_scope(scope, reuse=reuse):
            x = activation_fn(slim.conv2d(x, channel, 4, 2, activation_fn=activation_fn, scope='conv_0'))
            for i in range(1, self.n_dis) :
                channel *= 2
                x = activation_fn(ops_UNIT.instance_norm(slim.conv2d(x, channel, 4, 2, activation_fn=activation_fn, scope='conv_'+str(i)),scope='ins_norm_'+str(i)))
            x = activation_fn(slim.conv2d(x, 1, 1, 1, activation_fn=None, scope='dis_logit'))
            return x
    # END of DISCRIMINATORS
    ##############################################################################

    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)

        out_A = self.generator(out, self.is_training, scope="generator_A")
        out_B = self.generator(out, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_A")

        return out, shared

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        # self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba)
            x_aba, shared_aba = self.generate_b2a(x_ab)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test
            self.fake_A, _ = self.generate_b2a(domain_B) # for test

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()

    def init_net(self, args):
        # if args.pretrained_path is not None:
        #     var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='LinkNet')
        #     var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
        #     self.saverPart = tf.train.Saver(var, max_to_keep=5)
        if args.pretrained_vgg_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            var = var_filter_by_exclude(var, exclude_scopes=['UNIT/discriminator'])            
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        # if args.pretrained_path is not None:
        #     self.saverPart.restore(self.sess, args.pretrained_path)
        #     print('restored from pretrained_path:', args.pretrained_path)
        # elif self.ckpt_path is not None:
        #     self.saver.restore(self.sess, self.ckpt_path)
        #     print('restored from ckpt_path:', self.ckpt_path)

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)
        # if args.pretrained_path is not None:
        #     self.saverPart.restore(self.sess, args.pretrained_path)
        #     print('restored from pretrained_path:', args.pretrained_path)

        counter = args.global_step
        start_time = time.time()


        if 0==args.global_step:
            if args.continue_train and self.load_last_ckpt(args.model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            ## global_step is set manually
            if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # batch_idxs = self.sample_num // self.batch_size
            batch_idxs = self.sample_num
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            # idx = int(epoch/len(self.weights_schedule))
            # loss_weights = self.weights_schedule[idx]

            for idx in range(0, batch_idxs):
                step_ph = epoch*batch_idxs + idx
                num_steps = args.epoch*batch_idxs
                lr = args.lr*((1 - counter / num_steps)**0.9)
                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                # Update G
                for g_iter in range(args.G_update):
                    fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], \
                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, x_aa, x_ba, x_ab, x_bb = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, self.x_aa, self.x_ba, self.x_ab, self.x_bb], \
            feed_dict={self.is_training : False}
        )
        real_A = unprocess_image(real_A, 127.5, 127.5)
        real_B = unprocess_image(real_B, 127.5, 127.5)
        fake_A = unprocess_image(fake_A, 127.5, 127.5)
        fake_B = unprocess_image(fake_B, 127.5, 127.5)
        x_aa = unprocess_image(x_aa, 127.5, 127.5)
        x_ba = unprocess_image(x_ba, 127.5, 127.5)
        x_ab = unprocess_image(x_ab, 127.5, 127.5)
        x_bb = unprocess_image(x_bb, 127.5, 127.5)
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A, [self.batch_size, 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B, [self.batch_size, 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A, [self.batch_size, 1],
                    '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B, [self.batch_size, 1],
                    '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

    def save(self, model_dir, step):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.saver.save(self.sess,
                        os.path.join(model_dir, self.model_name),
                        global_step=step)

    def load_last_ckpt(self, model_dir):
        print(" [*] Reading last checkpoint...")

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))
            return True
        else:
            return False

    def load_ckpt(self, model_dir, step):
        print(" [*] Reading checkpoint of step {}...".format(step))

        # ckpt_path = glob(os.path.join(model_dir, '*model-{}*'.format(step)))
        ckpt_path = os.path.join(model_dir, '{}-{}'.format(self.model_name, step))
        ckpt_path_backup = os.path.join(model_dir, 'backup_{}-{}'.format(self.model_name, step))
        if os.path.exists(ckpt_path+'.meta'):
            self.saver.restore(self.sess, ckpt_path)
            return True
        elif os.path.exists(ckpt_path_backup+'.meta'):
            self.saver.restore(self.sess, ckpt_path_backup)
            return True
        else:
            return False

    def test(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        if args.pretrained_path is not None:
            self.saverPart.restore(self.sess, args.pretrained_path)
            print('restored from pretrained_path:', args.pretrained_path)

        if args.test_model_path is not None:
            self.saverTest.restore(self.sess, args.test_model_path)
            print('restored from test_model_path:', args.test_model_path)

        A_dir = os.path.join(args.test_dir+'_'+args.dataset_dir.split('/')[-1]+'_%dstyle_TrAsTs'%args.num_style, 'A')
        B_dir = os.path.join(args.test_dir+'_'+args.dataset_dir.split('/')[-1]+'_%dstyle_TrAsTs'%args.num_style, 'B')
        A2B_dir = os.path.join(args.test_dir+'_'+args.dataset_dir.split('/')[-1]+'_%dstyle_TrAsTs'%args.num_style, 'ab')
        B2A_dir = os.path.join(args.test_dir+'_'+args.dataset_dir.split('/')[-1]+'_%dstyle_TrAsTs'%args.num_style, 'ba')
        if not os.path.exists(A2B_dir):
            os.makedirs(A2B_dir)
        if not os.path.exists(B2A_dir):
            os.makedirs(B2A_dir)
        if not os.path.exists(A_dir):
            os.makedirs(A_dir)
        if not os.path.exists(B_dir):
            os.makedirs(B_dir)

        counter = 0
        start_time = time.time()

        # if self.load_ckpt(args.model_dir, args.global_step):
        #     print " [*] Load SUCCESS, args.global_step:%d"%args.global_step
        # else:
        #     raise Exception(" [!] Load failed...")

        if args.num_style>1:
            assert args.batch_size == 1
            batch_idxs = 1
            for idx in range(batch_idxs):
                counter += 1
                # self.sample_model(args.test_dir, epoch, idx)
                self.get_test_result_multi_style(A_dir, B_dir, A2B_dir, B2A_dir, args.num_style, args.save_test_dis_score)
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                          % (1, idx, batch_idxs, time.time() - start_time))
        else:
            for epoch in range(2):
                # pdb.set_trace()
                batch_idxs = int(math.ceil(self.sample_num/self.batch_size))
                for idx in range(0, batch_idxs):
                    counter += 1
                    # self.sample_model(args.test_dir, epoch, idx)
                    self.get_test_result(A_dir, B_dir, A2B_dir, B2A_dir, args.save_test_dis_score)
                    if np.mod(counter, args.print_freq) == 1:
                        # display training status
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                              % (epoch, idx, batch_idxs, time.time() - start_time))

    def get_test_result_multi_style(self, A_dir, B_dir, A2B_dir, B2A_dir, num_style, save_dis_score=False):
        img_name_A, img_name_B, real_A, real_B = [], [], [], []
        for i in range(num_style):
            img_name_A_one, img_name_B_one, real_A_one, real_B_one = self.sess.run(
                [self.img_name_A, self.img_name_B, self.real_A, self.real_B], \
                feed_dict={self.is_training : False}
            )
            img_name_A.append(img_name_A_one)
            img_name_B.append(img_name_B_one)
            real_A.append(real_A_one)
            real_B.append(real_B_one)

        combined = list(zip(img_name_A, img_name_B, real_A, real_B))
        combined = sorted(combined, key=lambda x: x[0])
        img_name_A[:], img_name_B[:], real_A[:], real_B[:] = zip(*combined)
        img_name_A = np.concatenate(img_name_A, axis=0)
        img_name_B = np.concatenate(img_name_B, axis=0)
        # pdb.set_trace()
        real_A = np.concatenate(real_A, axis=0)
        real_B = np.concatenate(real_B, axis=0)

        for i in range(num_style):
            for j in range(num_style):
                # pdb.set_trace()
                x_ba_tmp, x_ab_tmp = self.sess.run(
                    [self.x_ba, self.x_ab], \
                    feed_dict={self.is_training : True, 
                                self.real_A : np.tile(real_A[i,:,:,:], [1, 1, 1, 1]), 
                                self.real_B : np.tile(real_B[j,:,:,:], [1, 1, 1, 1])}
                    # feed_dict={self.is_training : False, 
                    #             self.real_A : np.tile(real_A[i,:,:,:], [1, 1, 1, 1]), 
                    #             self.real_B : np.tile(real_B[j,:,:,:], [1, 1, 1, 1])}
                )
                x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
                x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
                if save_dis_score:
                    scipy.misc.imsave('{}/{}_to_{}_{:04f}.png'.format(B2A_dir, img_name_B[j].split(".")[0], img_name_A[i].split(".")[0], d_loss_a), x_ba_tmp_img[0])
                    scipy.misc.imsave('{}/{}_to_{}_{:04f}.png'.format(A2B_dir, img_name_A[i].split(".")[0], img_name_B[j].split(".")[0], d_loss_b), x_ab_tmp_img[0])
                else:
                    scipy.misc.imsave('{}/{}_to_{}.png'.format(B2A_dir, img_name_B[j].split(".")[0], img_name_A[i].split(".")[0]), x_ba_tmp_img[0])
                    scipy.misc.imsave('{}/{}_to_{}.png'.format(A2B_dir, img_name_A[i].split(".")[0], img_name_B[j].split(".")[0]), x_ab_tmp_img[0])

        real_A = unprocess_image(real_A, 127.5, 127.5)
        real_B = unprocess_image(real_B, 127.5, 127.5)
        for i in range(img_name_A.shape[0]):
            scipy.misc.imsave('{}/{}.png'.format(A_dir, img_name_A[i].split(".")[0]), real_A[i,:,:,:])
            scipy.misc.imsave('{}/{}.png'.format(B_dir, img_name_B[i].split(".")[0]), real_B[i,:,:,:])


    def get_test_result(self, A_dir, B_dir, A2B_dir, B2A_dir, save_dis_score=False):
        img_name_A, img_name_B, real_A, real_B, x_ba, x_ab, d_loss_a, d_loss_b = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.x_ba, self.x_ab, self.D_loss_A, self.D_loss_B], \
            feed_dict={self.is_training : True}
            # feed_dict={self.is_training : False}
        )
        real_A = unprocess_image(real_A, 127.5, 127.5)
        real_B = unprocess_image(real_B, 127.5, 127.5)
        x_ba = unprocess_image(x_ba, 127.5, 127.5)
        x_ab = unprocess_image(x_ab, 127.5, 127.5)
        for i in range(img_name_A.shape[0]):
            scipy.misc.imsave('{}/{}.png'.format(A_dir, img_name_A[i].split(".")[0]), real_A[i,:,:,:])
            scipy.misc.imsave('{}/{}.png'.format(B_dir, img_name_B[i].split(".")[0]), real_B[i,:,:,:])
            if save_dis_score:
                scipy.misc.imsave('{}/{}_to_{}_{:04f}.png'.format(B2A_dir, img_name_B[i].split(".")[0], img_name_A[i].split(".")[0], d_loss_a), x_ba[i,:,:,:])
                scipy.misc.imsave('{}/{}_to_{}_{:04f}.png'.format(A2B_dir, img_name_A[i].split(".")[0], img_name_B[i].split(".")[0], d_loss_b), x_ab[i,:,:,:])
            else:
                scipy.misc.imsave('{}/{}_to_{}.png'.format(B2A_dir, img_name_B[i].split(".")[0], img_name_A[i].split(".")[0]), x_ba[i,:,:,:])
                scipy.misc.imsave('{}/{}_to_{}.png'.format(A2B_dir, img_name_A[i].split(".")[0], img_name_B[i].split(".")[0]), x_ab[i,:,:,:])

    ## TODO
    # def test(self):
    #     tf.global_variables_initializer().run()
    #     test_A_files = glob('{}/{}/*.*'.format(self.data_parent_dir, self.dataset_dir + '/testA'))
    #     test_B_files = glob('{}/{}/*.*'.format(self.data_parent_dir, self.dataset_dir + '/testB'))

    #     """
    #     testA, testB = test_data(dataset_name=self.dataset_name, size=self.img_size)
    #     test_A_images = testA[:]
    #     test_B_images = testB[:]
    #     """

    #     ## global_step is set manually
    #     if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")

    #     self.saver = tf.train.Saver()
    #     could_load, checkpoint_counter = self.load(self.model_dir)

    #     if could_load :
    #         print(" [*] Load SUCCESS")
    #     else :
    #         print(" [!] Load failed...")

    #     # write html for visual comparison
    #     index_path = os.path.join(self.test_dir, 'index.html')
    #     index = open(index_path, 'w')
    #     index.write("<html><body><table><tr>")
    #     index.write("<th>name</th><th>input</th><th>output</th></tr>")

    #     for sample_file  in test_A_files : # A -> B
    #         print('Processing A image: ' + sample_file)
    #         sample_image = np.asarray(load_test_data(sample_file))
    #         image_path = os.path.join(self.test_dir,'{0}'.format(os.path.basename(sample_file)))

    #         fake_img = self.sess.run(self.fake_B, feed_dict = {self.domain_A : sample_image, self.prob : 0.0, self.is_training : False})

    #         save_images(fake_img, [1, 1], image_path)
    #         index.write("<td>%s</td>" % os.path.basename(image_path))
    #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
    #             '..' + os.path.sep + sample_file), self.img_size, self.img_size))
    #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
    #             '..' + os.path.sep + image_path), self.img_size, self.img_size))
    #         index.write("</tr>")

    #     for sample_file  in test_B_files : # B -> A
    #         print('Processing B image: ' + sample_file)
    #         sample_image = np.asarray(load_test_data(sample_file))
    #         image_path = os.path.join(self.test_dir,'{0}'.format(os.path.basename(sample_file)))

    #         fake_img = self.sess.run(self.fake_A, feed_dict = {self.domain_B : sample_image, self.prob : 0.0, self.is_training : False})

    #         save_images(fake_img, [1, 1], image_path)
    #         index.write("<td>%s</td>" % os.path.basename(image_path))
    #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
    #             '..' + os.path.sep + sample_file), self.img_size, self.img_size))
    #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
    #             '..' + os.path.sep + image_path), self.img_size, self.img_size))
    #         index.write("</tr>")
    #     index.close()

    def _load_batch_data_tf(self, dataset, is_training=True):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        img_name_A, img_name_B, img_A, img_B, img_A_seg, img_B_seg, img_A_seg_class, img_B_seg_class, A_seg_valid, B_seg_valid  = data_provider.get([
            'image_name_A', 'image_name_B', 'image_raw_A', 'image_raw_B', 'image_raw_A_seg', 'image_raw_B_seg', 'image_raw_A_seg_class', 'image_raw_B_seg_class', 'A_seg_valid', 'B_seg_valid'])

        img_A = tf.reshape(img_A, [self.img_h_original, self.img_w_original, 3])        
        img_B = tf.reshape(img_B, [self.img_h_original, self.img_w_original, 3])
        img_A_seg = tf.reshape(img_A_seg, [self.img_h_original, self.img_w_original, 3])
        img_B_seg = tf.reshape(img_B_seg, [self.img_h_original, self.img_w_original, 3])
        # img_A_seg_class = tf.reshape(img_A_seg_class, [int(self.img_h_original/8), int(self.img_w_original/8), self.segment_class])
        # img_B_seg_class = tf.reshape(img_B_seg_class, [int(self.img_h_original/8), int(self.img_w_original/8), self.segment_class])
        # img_A_seg_class = tf.to_float(img_A_seg_class)
        # img_B_seg_class = tf.to_float(img_B_seg_class)
        img_A_seg_class = tf.reshape(img_A_seg_class, [self.img_h_original, self.img_w_original, 1]) ## [batch, h, w]
        img_B_seg_class = tf.reshape(img_B_seg_class, [self.img_h_original, self.img_w_original, 1])
        ## Perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        # img_A_seg_class = tf.one_hot(img_A_seg_class, self.segment_class, axis=-1)  ## [batch, h, w, class]
        # img_B_seg_class = tf.one_hot(img_B_seg_class, self.segment_class, axis=-1)

        img_A = process_image(tf.to_float(img_A), 127.5, 127.5)
        img_B = process_image(tf.to_float(img_B), 127.5, 127.5)
        img_A_seg = process_image(tf.to_float(img_A_seg), 127.5, 127.5)
        img_B_seg = process_image(tf.to_float(img_B_seg), 127.5, 127.5)

        imgs_name_A, imgs_name_B, imgs_A, imgs_B, imgs_A_seg, imgs_B_seg, imgs_A_seg_class, imgs_B_seg_class, A_seg_valids, B_seg_valids = tf.train.batch(
                [img_name_A, img_name_B, img_A, img_B, img_A_seg, img_A_seg, img_A_seg_class, img_B_seg_class, A_seg_valid, B_seg_valid], 
                batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        if is_training:
            color_prob=0.5 if self.color_aug else 0.0
            imgs_A, imgs_A_seg, imgs_A_seg_class, imgs_A_depth, _ = data_augment(
                    rgb=imgs_A, # 3 channels
                    seg=imgs_A_seg, # 3 channels
                    seg_class_map=imgs_A_seg_class, # seg_class channels
                    resize=[self.img_h, self.img_w], # (width, height) tuple or None
                    horizontal_flip=True,
                    crop_probability=0.5, # How often we do crops
                    color_probability=color_prob)  # How often we do color jitter

            imgs_B, imgs_B_seg, imgs_B_seg_class, imgs_B_depth, _ = data_augment(
                    rgb=imgs_B, # 3 channels
                    seg=imgs_B_seg, # 3 channels
                    seg_class_map=imgs_B_seg_class, # seg_class channels
                    resize=[self.img_h, self.img_w], # (width, height) tuple or None
                    horizontal_flip=True,
                    crop_probability=0.5) # How often we do crops
                    # color_probability=color_prob)  # How often we do color jitter
        else:
            imgs_A, imgs_A_seg, imgs_A_seg_class, imgs_A_depth, _ = data_augment(
                    rgb=imgs_A, # 3 channels
                    seg=imgs_A_seg, # 3 channels
                    seg_class_map=imgs_A_seg_class, # 1 channel
                    resize=[self.img_h, self.img_w]) # (width, height) tuple or None

            imgs_B, imgs_B_seg, imgs_B_seg_class, imgs_B_depth, _ = data_augment(
                    rgb=imgs_B, # 3 channels
                    seg=imgs_B_seg, # 3 channels
                    seg_class_map=imgs_B_seg_class, # 1 channel
                    resize=[self.img_h, self.img_w]) # (width, height) tuple or None

        imgs_AB = tf.concat([imgs_A, imgs_B], axis=-1)
        imgs_AB_seg = tf.concat([imgs_A_seg, imgs_B_seg], axis=-1)
        imgs_A_seg_class = tf.squeeze(imgs_A_seg_class, axis=-1)
        imgs_B_seg_class = tf.squeeze(imgs_B_seg_class, axis=-1)
        return imgs_name_A, imgs_name_B, imgs_AB, imgs_AB_seg, imgs_A_seg_class, imgs_B_seg_class, A_seg_valids, B_seg_valids

    def _load_batch_data_tf_noSeg(self, dataset, is_training=True):
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)

        img_name_A, img_name_B, img_A, img_B = data_provider.get([
            'image_name_A', 'image_name_B', 'image_raw_A', 'image_raw_B'])

        img_A = tf.reshape(img_A, [self.img_h_original, self.img_w_original, 3])        
        img_B = tf.reshape(img_B, [self.img_h_original, self.img_w_original, 3])

        img_A = process_image(tf.to_float(img_A), 127.5, 127.5)
        img_B = process_image(tf.to_float(img_B), 127.5, 127.5)

        imgs_name_A, imgs_name_B, imgs_A, imgs_B = tf.train.batch(
                [img_name_A, img_name_B, img_A, img_B], 
                batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        if is_training:
            color_prob=0.5 if self.color_aug else 0.0
            crop_probability = 0.5
            # horizontal_flip = False
            horizontal_flip = True
            imgs_A, _, _, _, _ = data_augment(
                    rgb=imgs_A, # 3 channels
                    resize=[self.img_h, self.img_w], # (width, height) tuple or None
                    horizontal_flip=horizontal_flip,
                    crop_probability=crop_probability, # How often we do crops
                    color_probability=color_prob)  # How often we do color jitter

            imgs_B, _, _, _, _ = data_augment(
                    rgb=imgs_B, # 3 channels
                    resize=[self.img_h, self.img_w], # (width, height) tuple or None
                    horizontal_flip=horizontal_flip,
                    crop_probability=crop_probability) # How often we do crops
                    # color_probability=color_prob)  # How often we do color jitter
        else:
            imgs_A, _, _, _, _ = data_augment(
                    rgb=imgs_A, # 3 channels
                    resize=[self.img_h, self.img_w]) # (width, height) tuple or None

            imgs_B, _, _, _, _ = data_augment(
                    rgb=imgs_B, # 3 channels
                    resize=[self.img_h, self.img_w]) # (width, height) tuple or None

        imgs_AB = tf.concat([imgs_A, imgs_B], axis=-1)
        return imgs_name_A, imgs_name_B, imgs_AB


class UNIT_VAEGAN_recon(UNIT):
    def VAEGAN_recon(self, x, is_training, scope='VAEGAN_recon', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            enc = self.encoder(x, is_training, scope="part1")
            latent = self.share_encoder(enc, is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            out = self.generator(out, is_training, scope="part4")

        return out, latent

    def _build_model(self):
        self._define_input()
        # self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        self.x_aa, latent_A = self.VAEGAN_recon(domain_A, self.is_training, scope='VAEGAN_recon_A')
        self.x_bb, latent_B = self.VAEGAN_recon(domain_B, self.is_training, scope='VAEGAN_recon_B')

        real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
        real_B_logit = self.discriminator(domain_B, scope="discriminator_B")
        fake_A_logit = self.discriminator(self.x_aa, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(self.x_bb, reuse=True, scope="discriminator_B")

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss_a = ops_UNIT.KL_divergence(latent_A)
        enc_loss_b = ops_UNIT.KL_divergence(latent_B)

        l1_loss_a = ops_UNIT.L1_loss(self.x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(self.x_bb, domain_B) # identity

        # Generator_A_loss = self.L1_weight * l1_loss_a + \
        #                    self.KL_weight * enc_loss_a

        # Generator_B_loss = self.L1_weight * l1_loss_b + \
        #                    self.KL_weight * enc_loss_b

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.KL_weight * enc_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.KL_weight * enc_loss_b

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'VAEGAN_recon' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.enc_loss_a_sum = tf.summary.scalar("KL_enc_loss_a", enc_loss_a)
        self.enc_loss_b_sum = tf.summary.scalar("KL_enc_loss_b", enc_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.enc_loss_a_sum, self.enc_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_path is not None:
            self.saverPart.restore(self.sess, args.pretrained_path)
            print('restored from pretrained_path:', args.pretrained_path)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

        counter = args.global_step
        start_time = time.time()


        if 0==args.global_step:
            if args.continue_train and self.load_last_ckpt(args.model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            ## global_step is set manually
            if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # batch_idxs = self.sample_num // self.batch_size
            batch_idxs = self.sample_num
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            # idx = int(epoch/len(self.weights_schedule))
            # loss_weights = self.weights_schedule[idx]

            for idx in range(0, batch_idxs):
                step_ph = epoch*batch_idxs + idx
                num_steps = args.epoch*batch_idxs
                lr = args.lr*((1 - counter / num_steps)**0.9)
                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                # Update G
                for g_iter in range(args.G_update):
                    _, g_loss, summary_str = self.sess.run([self.G_optim, self.Generator_loss, self.G_loss], \
                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, x_aa, x_bb = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.x_aa, self.x_bb], \
            feed_dict={self.is_training : False}
        )
        real_A = unprocess_image(real_A, 127.5, 127.5)
        real_B = unprocess_image(real_B, 127.5, 127.5)
        x_aa = unprocess_image(x_aa, 127.5, 127.5)
        x_bb = unprocess_image(x_bb, 127.5, 127.5)
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A, [real_A.shape[0], 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B, [real_B.shape[0], 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa, [x_aa.shape[0], 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb, [x_bb.shape[0], 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(UNIT):
    def init_net(self, args):
        assert args.pretrained_path is not None
        if args.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A') \
                + tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_B')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverPart = tf.train.Saver(var, max_to_keep=5)

        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def encoder_spec(self, x, is_training=True, reuse=False, scope="part1"):
        channel = self.ngf
        feaMap_list, gamma_list, beta_list = [], [], []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                feaMap_list.append(x)
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)

            return feaMap_list, gamma_list, beta_list

    def ins_specific_branch(self, x, is_training=False, scope='VAEGAN_recon', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            feaMap_list, gamma_list, beta_list = self.encoder_spec(x, is_training, scope="part1")
        return feaMap_list, gamma_list, beta_list

    def apply_feaMap_mask(self, x, feaMap):
        # mask = (tf.sign(feaMap)+1.0)/2.0 ## select the >0 elements as mask
        mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
        # mask = mask/2. - 1 ## norm [0.5, 1] to [0, 1]
        mask = mask/2. + 1./2. ## norm [0.5, 1] to [0.75, 1]
        # mask = mask/5.*2. + 0.6 ## norm [0.5, 1] to [0.8, 1]
        # mask = mask/4. + 3./4. ## norm [0.5, 1] to [0.875, 1]
        # mask = tf.ones_like(mask) ## norm [0.5, 1] to [1, 1]
        mask = tf.image.resize_images(mask, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        x = x * mask
        return x

    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)

                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out, shared

    def vgg_net(self, x, is_training=False, reuse=False, scope="style_content_vgg"):
        vgg = tensorflow_vgg19.Vgg19_style('./weights/vgg19.npy')
        with tf.variable_scope(scope, reuse=reuse):
            ## x should be rgb image [batch, height, width, 3] values scaled [0, 1]
            x = (x+1.0)/2.0 ## preprocess
            vgg.build(x)
            fea_list = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
            ## L2 norm by channel
            try:
                fea_list = [tf.nn.l2_normalize(fea, axis=[1,2])for fea in fea_list]
            except:
                fea_list = [tf.nn.l2_normalize(fea, dim=[1,2])for fea in fea_list]
        return fea_list

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE):
        x = tf.concat([x_A, x_B, x_ab, x_ba], axis=0)
        # feaMap_list, gamma_list, beta_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list_A = [tf.split(feaMap, 4, axis=0)[0] for feaMap in feaMap_list]
        feaMap_list_B = [tf.split(feaMap, 4, axis=0)[1] for feaMap in feaMap_list]
        feaMap_list_ab = [tf.split(feaMap, 4, axis=0)[2] for feaMap in feaMap_list]
        feaMap_list_ba = [tf.split(feaMap, 4, axis=0)[3] for feaMap in feaMap_list]

        num = len(feaMap_list_A)
        ## Use source domain ecnoder to extract the content feature map
        content_loss = 0.
        for i in range(num):
        # for i in range(2,num):
            # content_loss += ((float(i+1)/float(num))**2)*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_A[i], feaMap_list_ab[i])) \
            #             + ((float(i+1)/float(num))**2)*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_B[i], feaMap_list_ba[i]))
            # with tf.variable_scope('content_loss', reuse=tf.AUTO_REUSE):
            if self.content_loss_IN:

                # feaMap_list_A[i] = tf.Print(feaMap_list_A[i], [ops_UNIT.self_ins_norm_2d(feaMap_list_A[i])*1e-8- ops_UNIT.self_ins_norm_2d(feaMap_list_ab[i])*1e-8], summarize=40, message="fea_list[%d] is:"%i)
                content_loss += (float(i+1)/float(num))*ops_UNIT.L1_loss(ops_UNIT.instance_norm(feaMap_list_A[i], 'content_loss_%d'%i, False, tf.AUTO_REUSE), ops_UNIT.instance_norm(feaMap_list_ab[i], 'content_loss_%d'%i, False, tf.AUTO_REUSE)) \
                            + (float(i+1)/float(num))*ops_UNIT.L1_loss(ops_UNIT.instance_norm(feaMap_list_B[i], 'content_loss_%d'%i, False, tf.AUTO_REUSE), ops_UNIT.instance_norm(feaMap_list_ba[i], 'content_loss_%d'%i, False, tf.AUTO_REUSE))
                # content_loss = tf.Print(content_loss, [content_loss], summarize=40, message="content_loss is:")
            else:
                content_loss += (float(i+1)/float(num))*ops_UNIT.L1_loss(feaMap_list_A[i], feaMap_list_ab[i]) \
                            + (float(i+1)/float(num))*ops_UNIT.L1_loss(feaMap_list_B[i], feaMap_list_ba[i])

        ## Use target domain ecnoder to extract the style stastics, i.e. mean and var
        style_loss = 0.
        for i in range(num):
            style_loss += ops_UNIT.L1_loss(ops_UNIT.gram_matrix(feaMap_list_ab[i]), ops_UNIT.gram_matrix(feaMap_list_B[i]))
            style_loss += ops_UNIT.L1_loss(ops_UNIT.gram_matrix(feaMap_list_ba[i]), ops_UNIT.gram_matrix(feaMap_list_A[i]))

        return content_loss, style_loss

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _define_output(self):
        pass

    def _build_model(self):
        self._define_input()        
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + self.style_weight*style_loss + self.content_weight*content_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        self.D_loss_A = Discriminator_A_loss
        self.D_loss_B = Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_path is not None:
            self.saverPart.restore(self.sess, args.pretrained_path)
            print('restored from pretrained_path:', args.pretrained_path)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

        counter = args.global_step
        start_time = time.time()

        if 0==args.global_step:
            if args.continue_train and self.load_last_ckpt(args.model_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            ## global_step is set manually
            if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # batch_idxs = self.sample_num // self.batch_size
            batch_idxs = self.sample_num

            d_loss = np.inf
            for idx in range(0, batch_idxs):
                step_ph = epoch*batch_idxs + idx
                num_steps = args.epoch*batch_idxs
                lr = args.lr*((1 - counter / num_steps)**0.9)
                # Update D
                # if d_loss>3.0:
                if self.D_optim is not None:
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
                                                            feed_dict={self.is_training : True, self.lr : lr})
                    self.writer.add_summary(summary_str, counter)

                # Update G
                for g_iter in range(args.G_update):
                    fake_A, fake_B, _, g_loss, content_loss, style_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.content_loss, self.style_loss, self.G_loss], \
                                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f, content_loss: %.8f, style_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, content_loss, style_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, feaMapA, feaMapB = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.feaMask_list_A[0], self.feaMask_list_B[0]], \
            feed_dict={self.is_training : False}
        )

        real_A_img = unprocess_image(real_A, 127.5, 127.5)
        real_B_img = unprocess_image(real_B, 127.5, 127.5)
        fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
        fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
        fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
        fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
        fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
        fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
        x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
        x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
        x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
        x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
        img_name_A = img_name_A[0]
        img_name_B = img_name_B[0]
        save_images(real_A_img, [self.batch_size, 1],
                    '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(real_B_img, [self.batch_size, 1],
                    '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_img, [self.batch_size, 1],
                    '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_img, [self.batch_size, 1],
                    '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_feaMasked_img, [self.batch_size, 1],
                    '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(fake_A_insNormed_img, [self.batch_size, 1],
                    '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(fake_B_insNormed_img, [self.batch_size, 1],
                    '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_aa_img, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_img, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_img, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_img, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
## TODO: save mask ##
        b,h,w,c = feaMapA.shape
        # pdb.set_trace()
        for b_idx in range(b):
            mapA = np.tile(feaMapA[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            mapB = np.tile(feaMapB[b_idx,:,:,:].transpose([2,0,1]).reshape([c,h,w,1])*255., [1,1,1,3])
            save_images(mapA, [16, int(mapA.shape[0]/16)],
                        '{}/feaMapA0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_A.split(".")[0]))
            save_images(mapB, [16, int(mapB.shape[0]/16)],
                        '{}/feaMapB0_{:02d}_{:02d}_{:04d}_{}.png'.format(sample_dir, b_idx, epoch, idx, img_name_B.split(".")[0]))

        for i in range(self.batch_size):
            # pdb.set_trace()
            fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
                [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
                feed_dict={self.is_training : False, 
                            self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
                            self.real_B : real_B}
            )
            fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
            fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
            x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
            x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
            save_images(fake_A_tmp_img, [self.batch_size, 1],
                        '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(fake_B_tmp_img, [self.batch_size, 1],
                        '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ba_tmp_img, [self.batch_size, 1],
                        '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            save_images(x_ab_tmp_img, [self.batch_size, 1],
                        '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            
