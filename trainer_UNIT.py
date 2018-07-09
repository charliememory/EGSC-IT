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
from datasets import gta, synsf, dataset_utils, celeba

import ops_UNIT
from labels_utils import *
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

                
class UNIT_FeaLossDis(UNIT):
    ##############################################################################
    # BEGIN of DISCRIMINATORS
    def discriminator(self, x, reuse=False, scope="discriminator"):
        channel = self.ndf
        with tf.variable_scope(scope, reuse=reuse):
            fea_list = []
            x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')
            fea_list.append(x)
            for i in range(1, self.n_dis) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                fea_list.append(x)
                channel *= 2

            x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')
            fea_list.append(x)
            return x, fea_list
    # END of DISCRIMINATORS
    ##############################################################################

    def _define_input(self):
        ## Input param
        # self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba)
            x_aba, shared_aba = self.generate_b2a(x_ab)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)        
            real_A_logit, real_A_fea_list = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit, real_B_fea_list = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit, fake_A_fea_list = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit, fake_B_fea_list = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit, fake_A_fea_list = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit, fake_B_fea_list = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test
            self.fake_A, _ = self.generate_b2a(domain_B) # for test

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_fea_loss_a = tf.reduce_mean([ops_UNIT.L1_loss(real_A_fea_list[i], fake_A_fea_list[i]) for i in range(len(real_A_fea_list))])
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_fea_loss_b = tf.reduce_mean([ops_UNIT.L1_loss(real_B_fea_list[i], fake_B_fea_list[i]) for i in range(len(real_B_fea_list))])

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
                           self.KL_cycle_weight * enc_bab_loss +\
                           1.0*G_fea_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss +\
                           1.0*G_fea_loss_b

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
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.G_fea_loss_a_sum = tf.summary.scalar("G_fea_loss_a", G_fea_loss_a)
        self.G_fea_loss_b_sum = tf.summary.scalar("G_fea_loss_b", G_fea_loss_b)
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
                        self.G_fea_loss_a_sum, self.G_fea_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_MultiSpecificBranchFromImg_Cycle(UNIT):
    def ins_specific_branch(self, x, is_training=True, reuse=False, scope="specific_encoder"):
        out_channel = self.ngf * pow(2, self.n_encoder-1)
        channel = self.ngf
        gamma_list, beta_list = [], []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2
            for i in range(0, self.n_gen_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv2_'+str(i))
                # channel *= 2
                gamma = ops_UNIT.fc(x, channel, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='fc_gamma_'+str(i))
                beta = ops_UNIT.fc(x, channel, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='fc_beta_'+str(i))
                gamma_list.append(gamma)
                beta_list.append(beta)
            return gamma_list, beta_list

    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, gamma_list=None, beta_list=None, fore_half=False, post_half=False, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                if i<=0:
                    if gamma_list is not None and beta_list is not None:
                        if fore_half or post_half:
                            assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                            x1, x2 = tf.split(x, 2, axis=0)
                            if fore_half:
                                x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma_list[-1-i], beta_list[-1-i])
                            if post_half:
                                x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma_list[-1-i], beta_list[-1-i])
                            x = tf.concat([x1,x2], axis=0)
                        else:
                            ## for generator_a2b and generator_b2a
                            assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                            x = ops_UNIT.apply_ins_norm_2d(x, gamma_list[-1-i], beta_list[-1-i])
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

    def translation(self, x_A, x_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch
        gamma_list_A, beta_list_A = self.ins_specific_branch(x_A, self.is_training, scope="specific_encoder_A")
        gamma_list_B, beta_list_B = self.ins_specific_branch(x_B, self.is_training, scope="specific_encoder_B")

        out_A = self.generator(out, self.is_training, gamma_list_A, beta_list_A, fore_half=True, post_half=True, scope="generator_A")
        # out_B = self.generator(out, self.is_training, gamma_list_A, beta_list_A, fore_half=True, post_half=True, scope="generator_B")
        out_B = self.generator(out, self.is_training, gamma_list_B, beta_list_B, fore_half=True, post_half=True, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B

    def generate_a2b(self, x_A, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, gamma_list, beta_list, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, gamma_list, beta_list, reuse=True, scope="generator_A")

        return out, shared

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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
        
    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, x_aa, x_ba, x_ab, x_bb = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, self.x_aa, self.x_ba, self.x_ab, self.x_bb], \
            feed_dict={self.is_training : False}
        )
        real_A_img = unprocess_image(real_A, 127.5, 127.5)
        real_B_img = unprocess_image(real_B, 127.5, 127.5)
        fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
        fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
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
        save_images(x_aa_img, [self.batch_size, 1],
                    '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_img, [self.batch_size, 1],
                    '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_img, [self.batch_size, 1],
                    '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_img, [self.batch_size, 1],
                    '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

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
            


class UNIT_MultiSpecificBranchFromImg_Cycle_simple(UNIT_MultiSpecificBranchFromImg_Cycle):
    def ins_specific_branch(self, x, is_training=True, reuse=False, scope="specific_encoder"):
        out_channel = self.ngf * pow(2, self.n_encoder-1)
        channel = self.ngf
        gamma_list, beta_list = [], []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            x = tf.image.resize_images(x, [pow(2, self.n_gen_resblock),pow(2, self.n_gen_resblock)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            for i in range(0, self.n_gen_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv2_'+str(i))
                # channel *= 2
                gamma = ops_UNIT.fc(x, channel, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='fc_gamma_'+str(i))
                beta = ops_UNIT.fc(x, channel, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='fc_beta_'+str(i))
                gamma_list.append(gamma)
                beta_list.append(beta)
            return gamma_list, beta_list


class UNIT_MultiSpecificBranchFromImg_Cycle_FCN(UNIT_MultiSpecificBranchFromImg_Cycle):
    def ins_specific_branch(self, x, is_training=True, reuse=False, scope="specific_encoder"):
        out_channel = self.ngf * pow(2, self.n_encoder-1)
        channel = self.ngf
        gamma_list, beta_list = [], []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                channel *= 2

            for i in range(0, self.n_gen_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                # channel *= 2
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)
            return gamma_list, beta_list


class UNIT_MultiSpecificBranchFromImg_Cycle_simple_GradSmooth(UNIT_MultiSpecificBranchFromImg_Cycle_simple):
    def gradient_map(self, x):
        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        _, _, _, ch = x.get_shape().as_list()
        kernels = []
        kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), ch) )
        kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), ch) )
        kernel = tf.constant(np.stack(kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        gradient_map = tf.abs(tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding="VALID"))
        return gradient_map

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm
            self.grad_fake_B = self.gradient_map(self.fake_B)
            self.grad_fake_A = self.gradient_map(self.fake_A)

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
        self.grad_loss = tf.reduce_mean(self.grad_fake_B) + tf.reduce_mean(self.grad_fake_A)

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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1.0*self.grad_loss
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

        self.grad_loss_sum = tf.summary.scalar("grad_loss", self.grad_loss)
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.grad_loss_sum])
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
                    fake_A, fake_B, _, g_loss, grad_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.grad_loss, self.G_loss], \
                                                        feed_dict={self.is_training : True, self.lr : lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                if np.mod(counter, args.print_freq) == 1:
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f, grad_loss: %.8f" \
                          % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss, grad_loss))
                    self.sample_model(args.sample_dir, epoch, counter)

                if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
                    self.save(args.model_dir, counter)


class UNIT_MultiSpecificBranchFromImg_Cycle_simple_GaussianSmooth(UNIT_MultiSpecificBranchFromImg_Cycle_simple):
    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm
            self.blur_fake_B = Smoother({'data':self.fake_B}, filter_size=13, sigma=2.0).get_output()
            self.blur_fake_A = Smoother({'data':self.fake_A}, filter_size=13, sigma=2.0).get_output()

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
        grad_loss_a = ops_UNIT.L1_loss(self.fake_B, self.blur_fake_B)
        grad_loss_b = ops_UNIT.L1_loss(self.fake_A, self.blur_fake_A)
        self.grad_loss = grad_loss_a + grad_loss_b

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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 100*self.grad_loss
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

        self.grad_loss_a_sum = tf.summary.scalar("grad_loss_a", grad_loss_a)
        self.grad_loss_b_sum = tf.summary.scalar("grad_loss_b", grad_loss_b)
        self.grad_loss_sum = tf.summary.scalar("grad_loss", self.grad_loss)
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.grad_loss_sum, self.grad_loss_a_sum, self.grad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()

class UNIT_MultiSpecificBranchFromImg_Cycle_simple_tvSmooth(UNIT_MultiSpecificBranchFromImg_Cycle_simple):
    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm
            self.blur_fake_B = tf.reduce_mean(tf.image.total_variation(self.fake_B))/self.img_h/self.img_w
            self.blur_fake_A = tf.reduce_mean(tf.image.total_variation(self.fake_A))/self.img_h/self.img_w

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
        grad_loss_a = ops_UNIT.L1_loss(self.fake_B, self.blur_fake_B)
        grad_loss_b = ops_UNIT.L1_loss(self.fake_A, self.blur_fake_A)
        self.grad_loss = grad_loss_a + grad_loss_b

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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 10.0*self.grad_loss
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

        self.grad_loss_a_sum = tf.summary.scalar("grad_loss_a", grad_loss_a)
        self.grad_loss_b_sum = tf.summary.scalar("grad_loss_b", grad_loss_b)
        self.grad_loss_sum = tf.summary.scalar("grad_loss", self.grad_loss)
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.grad_loss_sum, self.grad_loss_a_sum, self.grad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_MultiSpecificBranchFromImg_simple(UNIT_MultiSpecificBranchFromImg_Cycle_simple):
    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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


import nets.vgg
import tensorflow_vgg.vgg19
class UNIT_VggSpecificBranchFromImg_Cycle(UNIT_MultiSpecificBranchFromImg_Cycle):
    def init_net(self, args):
        assert args.pretrained_vgg_path is not None
        if args.pretrained_vgg_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
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

    def ins_specific_branch(self, x, is_training=False, reuse=False):
        _, end_points = nets.vgg.vgg_19_style(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels

        gamma_list, beta_list = [], []
        # for i in [1, 2, 3, 4]:
        for i in [3]: 
            fea = end_points['vgg_19/conv%d/conv%d_1'%(i,i)]
            mean, var = tf.nn.moments(fea, [1,2])
            gamma = mean - 1.
            beta = var
            gamma_list.append(gamma)
            beta_list.append(beta)
        
        return gamma_list, beta_list

    def translation(self, x_A, x_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator(out, self.is_training, gamma_list_A, beta_list_A, fore_half=True, post_half=True, scope="generator_A")
        out_B = self.generator(out, self.is_training, gamma_list_B, beta_list_B, fore_half=True, post_half=True, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

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

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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
        self._define_output()

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
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

class UNIT_VggSpecificBranchFromImg_Cycle_VggLoss(UNIT_VggSpecificBranchFromImg_Cycle):
    def vgg_loss(self, pred, gt, is_training=False, reuse=False):
        x = tf.concat([pred, gt], axis=0)
        _, end_points = nets.vgg.vgg_19_style(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels

        loss = 0.0
        for i in [1, 2, 3, 4]:
            fea = end_points['vgg_19/conv%d/conv%d_1'%(i,i)]
            mean, var = tf.nn.moments(fea, [1,2])
            mean_pred, mean_gt =  tf.split(mean, 2, axis=0)
            var_pred, var_gt = tf.split(var, 2, axis=0)
            loss += ops_UNIT.L2_loss(mean_pred, mean_gt) + ops_UNIT.L2_loss(var_pred, var_gt)
        
        return loss

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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

        vgg_loss_a = self.vgg_loss(x_ba, domain_A, is_training=False, reuse=tf.AUTO_REUSE)
        vgg_loss_b = self.vgg_loss(x_ab, domain_B, is_training=False, reuse=tf.AUTO_REUSE)

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss + \
                           0.0001*vgg_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss + \
                           0.0001*vgg_loss_b

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
        self.vgg_loss_a_sum = tf.summary.scalar("vgg_loss_a", vgg_loss_a)
        self.vgg_loss_b_sum = tf.summary.scalar("vgg_loss_b", vgg_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.vgg_loss_a_sum, self.vgg_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_VggSpecificBranchFromImg_Cycle_FeaLossDis(UNIT_VggSpecificBranchFromImg_Cycle):
    ##############################################################################
    # BEGIN of DISCRIMINATORS
    def discriminator(self, x, reuse=False, scope="discriminator"):
        channel = self.ndf
        with tf.variable_scope(scope, reuse=reuse):
            fea_list = []
            x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')
            fea_list.append(x)
            for i in range(1, self.n_dis) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                fea_list.append(x)
                channel *= 2

            x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')
            fea_list.append(x)
            return x, fea_list
    # END of DISCRIMINATORS
    ##############################################################################

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

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)        
            real_A_logit, real_A_fea_list = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit, real_B_fea_list = self.discriminator(domain_B, scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit, fake_A_fea_list = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit, fake_B_fea_list = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit, fake_A_fea_list = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit, fake_B_fea_list = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_fea_loss_a = tf.reduce_mean([ops_UNIT.L1_loss(real_A_fea_list[i], fake_A_fea_list[i]) for i in range(len(real_A_fea_list))])
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_fea_loss_b = tf.reduce_mean([ops_UNIT.L1_loss(real_B_fea_list[i], fake_B_fea_list[i]) for i in range(len(real_B_fea_list))])

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
                           self.KL_cycle_weight * enc_bab_loss +\
                           10.0*G_fea_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss +\
                           10.0*G_fea_loss_b

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
        self.G_fea_loss_a_sum = tf.summary.scalar("G_fea_loss_a", G_fea_loss_a)
        self.G_fea_loss_b_sum = tf.summary.scalar("G_fea_loss_b", G_fea_loss_b)
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
                        self.G_fea_loss_a_sum, self.G_fea_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes(UNIT_VggSpecificBranchFromImg_Cycle):
    def ins_specific_branch(self, x, is_training=False, reuse=False):
        _, end_points = nets.vgg.vgg_19_style(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
        layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
        # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
        gamma_list, beta_list = [], []
        # for i in [1, 2, 3, 4]:
        for i in [1, 2, 3]:
        # for i in [3]: 
            fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
            mean, var = tf.nn.moments(fea, [1,2])
            gamma = mean - 1.
            beta = var
            gamma_list.append(gamma)
            beta_list.append(beta)
        
        return gamma_list, beta_list

    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, gamma_list=None, beta_list=None, fore_half=False, post_half=False, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        channel = 256 ## the channel of VGG_conv_3_1 is 256
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                if i<=2:
                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        # channel_vgg = gamma.get_shape().as_list()[-1]
                        # x = ops_UNIT.conv(x, channel_vgg, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='convIns_in_'+str(i))
                        if fore_half or post_half:
                            assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                            x1, x2 = tf.split(x, 2, axis=0)
                            if fore_half:
                                x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                            if post_half:
                                x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                            x = tf.concat([x1,x2], axis=0)
                        else:
                            ## for generator_a2b and generator_b2a
                            assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                            x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)
                        # x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='convIns_out_'+str(i))


                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                x = ops_UNIT.conv(x, channel//2, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv2_'+str(i))
                channel = channel // 2

            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')
            # x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='deconv_out')

            return x
    # END of DECODERS
    ##############################################################################

class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_MultiDis(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")
            real_A_logit2 = self.discriminator(tf.image.resize_images(domain_A, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), scope="discriminator_A2")
            real_B_logit2 = self.discriminator(tf.image.resize_images(domain_B, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), scope="discriminator_B2")
            real_A_logit3 = self.discriminator(tf.image.resize_images(domain_A, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), scope="discriminator_A3")
            real_B_logit3 = self.discriminator(tf.image.resize_images(domain_B, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), scope="discriminator_B3")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
                fake_A_logit2 = self.discriminator(self.fake_A_pool.query(tf.image.resize_images(x_ba, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)), reuse=True, scope="discriminator_A2") # replay memory
                fake_B_logit2 = self.discriminator(self.fake_B_pool.query(tf.image.resize_images(x_ab, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)), reuse=True, scope="discriminator_B2") # replay memory
                fake_A_logit3 = self.discriminator(self.fake_A_pool.query(tf.image.resize_images(x_ba, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)), reuse=True, scope="discriminator_A3") # replay memory
                fake_B_logit3 = self.discriminator(self.fake_B_pool.query(tf.image.resize_images(x_ab, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)), reuse=True, scope="discriminator_B3") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")
                fake_A_logit2 = self.discriminator(tf.image.resize_images(x_ba, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), reuse=True, scope="discriminator_A2")
                fake_B_logit2 = self.discriminator(tf.image.resize_images(x_ab, [int(self.img_h/2), int(self.img_w/2)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), reuse=True, scope="discriminator_B2")
                fake_A_logit3 = self.discriminator(tf.image.resize_images(x_ba, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), reuse=True, scope="discriminator_A3")
                fake_B_logit3 = self.discriminator(tf.image.resize_images(x_ab, [int(self.img_h/4), int(self.img_w/4)], method=tf.image.ResizeMethod.BILINEAR, align_corners=True), reuse=True, scope="discriminator_B3")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = 1./6.*ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 1./6.*ops_UNIT.discriminator_loss(real_A_logit2, fake_A_logit2, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 1./6.*ops_UNIT.discriminator_loss(real_A_logit3, fake_A_logit3, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = 1./6.*ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 1./6.*ops_UNIT.discriminator_loss(real_B_logit2, fake_B_logit2, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 1./6.*ops_UNIT.discriminator_loss(real_B_logit3, fake_B_logit3, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

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
        self._define_output()


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggDis(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes):
    def ins_specific_branch(self, x, is_training=False, reuse=False):
        _, end_points = nets.vgg.vgg_19_style(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
        layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
        # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
        fea_list, gamma_list, beta_list = [], [], []
        # for i in [1, 2, 3, 4]:
        for i in [1, 2, 3]:
        # for i in [3]: 
            fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
            mean, var = tf.nn.moments(fea, [1,2])
            gamma = mean - 1.
            beta = var
            fea_list.append(fea)
            gamma_list.append(gamma)
            beta_list.append(beta)
        
        return fea_list, gamma_list, beta_list

    ##############################################################################
    # BEGIN of DISCRIMINATORS
    def feaDiscriminator(self, x, reuse=False, scope="discriminator"):
        channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=reuse):
            x = ops_UNIT.conv(x, channel, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_dis) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                # channel *= 2
                channel = max(1024, channel*2)

            x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='dis_logit')
            # x = ops_UNIT.conv(x, channels=1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='sigmoid', scope='dis_logit')

            return x
    # END of DISCRIMINATORS
    ##############################################################################

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        fea_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        fea_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        real_A_fea = fea_list_A[-1]
        real_B_fea = fea_list_B[-1]
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

        ## for fea-level adversarial training
        fea_list_A, _, _ = self.ins_specific_branch(x_ba, is_training=False, reuse=True)
        fea_list_B, _, _ = self.ins_specific_branch(x_ab, is_training=False, reuse=True)
        fake_A_fea = fea_list_A[-1]
        fake_B_fea = fea_list_B[-1]

        with tf.variable_scope('UNIT'):
            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
            real_B_logit = self.discriminator(domain_B, scope="discriminator_B")
            real_A_fea_logit = self.feaDiscriminator(real_A_fea, scope="fea_discriminator_A")
            real_B_fea_logit = self.feaDiscriminator(real_B_fea, scope="fea_discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
                fake_A_fea_logit = self.feaDiscriminator(self.fake_A_pool.query(fake_A_fea), reuse=True, scope="fea_discriminator_A") # replay memory
                fake_B_fea_logit = self.feaDiscriminator(self.fake_B_pool.query(fake_B_fea), reuse=True, scope="fea_discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")
                fake_A_fea_logit = self.feaDiscriminator(fake_A_fea, reuse=True, scope="fea_discriminator_A") # replay memory
                fake_B_fea_logit = self.feaDiscriminator(fake_B_fea, reuse=True, scope="fea_discriminator_B") # replay memory

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 0.5*ops_UNIT.discriminator_loss(real_A_fea_logit, fake_A_fea_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan) \
                    + 0.5*ops_UNIT.discriminator_loss(real_B_fea_logit, fake_B_fea_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

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
        self._define_output()


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes):
    def vgg_loss(self, pred, gt, is_training=False, reuse=False):
        x = tf.concat([pred, gt], axis=0)
        _, end_points = nets.vgg.vgg_19_style(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels

        loss = 0.0
        # for i in [1, 2, 3, 4]:
        for i in [1, 2, 3]:
            fea = end_points['vgg_19/conv%d/conv%d_1'%(i,i)]
            mean, var = tf.nn.moments(fea, [1,2])
            mean_pred, mean_gt =  tf.split(mean, 2, axis=0)
            var_pred, var_gt = tf.split(var, 2, axis=0)
            loss += ops_UNIT.L2_loss(mean_pred, mean_gt) + ops_UNIT.L2_loss(var_pred, var_gt)
        
        return loss

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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

        vgg_loss_a = self.vgg_loss(x_ba, domain_A, is_training=False, reuse=tf.AUTO_REUSE)
        vgg_loss_b = self.vgg_loss(x_ab, domain_B, is_training=False, reuse=tf.AUTO_REUSE)
        
        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss + \
                           0.0005*vgg_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss + \
                           0.0005*vgg_loss_b

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
        self.vgg_loss_a_sum = tf.summary.scalar("vgg_loss_a", vgg_loss_a)
        self.vgg_loss_b_sum = tf.summary.scalar("vgg_loss_b", vgg_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.vgg_loss_a_sum, self.vgg_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss_noGAN(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, reuse=False)
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, reuse=True)
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        # G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        # G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        # D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        # D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        vgg_loss_a = self.vgg_loss(x_ba, domain_A, is_training=False, reuse=tf.AUTO_REUSE)
        vgg_loss_b = self.vgg_loss(x_ab, domain_B, is_training=False, reuse=tf.AUTO_REUSE)
        
        Generator_A_loss = self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss + \
                           1.*vgg_loss_a

        Generator_B_loss = self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss + \
                           1.*vgg_loss_b

        # Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        # Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        # self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        # D_vars = [var for var in t_vars if 'discriminator' in var.name]

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        # self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)
        self.D_optim = None

        """" Summary """
        # self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        # self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
        self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
        self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
        self.vgg_loss_a_sum = tf.summary.scalar("vgg_loss_a", vgg_loss_a)
        self.vgg_loss_b_sum = tf.summary.scalar("vgg_loss_b", vgg_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        # self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        # self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        # self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.vgg_loss_a_sum, self.vgg_loss_b_sum])
        # self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_LAB(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes):
    def _define_output(self):        
        self.real_A = process_image(lab_to_rgb(deprocess_lab(self.real_A))*255., 127.5, 127.5)
        self.real_B = process_image(lab_to_rgb(deprocess_lab(self.real_B))*255., 127.5, 127.5)
        self.fake_A = process_image(lab_to_rgb(deprocess_lab(self.fake_A))*255., 127.5, 127.5)
        self.fake_A = process_image(lab_to_rgb(deprocess_lab(self.fake_A))*255., 127.5, 127.5)
        self.x_aa = process_image(lab_to_rgb(deprocess_lab(self.x_aa))*255., 127.5, 127.5)
        self.x_ba = process_image(lab_to_rgb(deprocess_lab(self.x_ba))*255., 127.5, 127.5)
        self.x_ab = process_image(lab_to_rgb(deprocess_lab(self.x_ab))*255., 127.5, 127.5)
        self.x_bb = process_image(lab_to_rgb(deprocess_lab(self.x_bb))*255., 127.5, 127.5)

        # self.x_ab = tf.Print(self.x_ab, [tf.reduce_max(self.x_ab),tf.reduce_min(self.x_ab)], summarize=40, message="self.x_ab max-min is:")

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

        # img_A = process_image(tf.to_float(img_A), 127.5, 127.5)
        # img_B = process_image(tf.to_float(img_B), 127.5, 127.5)
        # img_A_seg = process_image(tf.to_float(img_A_seg), 127.5, 127.5)
        # img_B_seg = process_image(tf.to_float(img_B_seg), 127.5, 127.5)
        img_A = tf.to_float(img_A)/255.
        img_B = tf.to_float(img_B)/255.
        img_A_seg = tf.to_float(img_A_seg)/255.
        img_B_seg = tf.to_float(img_B_seg)/255.

        # img_A = tf.Print(img_A, [tf.reduce_max(img_A), tf.reduce_min(img_A)], summarize=40, message="rgb img_A max-min is:")
        # img_A = rgb_to_lab(tf.to_float(img_A))
        # img_A = tf.Print(img_A, [tf.reduce_max(img_A), tf.reduce_min(img_A)], summarize=40, message="lab img_A max-min is:")
        # img_A = preprocess_lab(img_A)
        # img_A = tf.Print(img_A, [tf.reduce_max(img_A), tf.reduce_min(img_A)], summarize=40, message="lab norm img_A max-min is:")
        img_A = preprocess_lab(rgb_to_lab(tf.to_float(img_A)))
        img_B = preprocess_lab(rgb_to_lab(tf.to_float(img_B)))
        img_A_seg = preprocess_lab(rgb_to_lab(tf.to_float(img_A_seg)))
        img_B_seg = preprocess_lab(rgb_to_lab(tf.to_float(img_B_seg)))


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


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss_noGAN_LAB(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_LAB, UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_VggLoss_noGAN):
    pass


class UNIT_AdaIN(UNIT):
    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, gamma_list=None, beta_list=None, fore_half=False, post_half=False, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        if gamma_list is None:
            new_gamma_list = []
        else:
            new_gamma_list = gamma_list
        if beta_list is None:
            new_beta_list = []
        else:
            new_beta_list = beta_list

        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i==0:
                if i<3:
                    if gamma_list is not None and beta_list is not None:
                        ## used in def translation
                        if fore_half or post_half: 
                            assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                            x1, x2 = tf.split(x, 2, axis=0)
                            ## apply gamma_list and beta_list to fore_half of x
                            if fore_half: 
                                x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma_list[-1-i], beta_list[-1-i])
                            ## apply gamma_list and beta_list to post of x
                            if post_half: 
                                x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma_list[-1-i], beta_list[-1-i])
                            x = tf.concat([x1,x2], axis=0)
                        ## used in def generator_a2b generator_b2a
                        else: 
                            ## for generator_a2b and generator_b2a
                            assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                            x = ops_UNIT.apply_ins_norm_2d(x, gamma_list[-1-i], beta_list[-1-i])
                    else:
                        if (not fore_half) and (not post_half):
                            pass
                        ## apply norm param of post half to fore_half of x
                        elif fore_half: 
                            x_dst, x_ref = tf.split(x, 2, axis=0)
                            x_dst, gamma, beta = ops_UNIT.apply_ins_norm_2d_like(x_dst, x_ref)
                            new_gamma_list.append(gamma)
                            new_beta_list.append(beta)
                            x = tf.concat([x_dst, x_ref], axis=0)
                        ## apply norm param of fore_half to post half of x
                        elif post_half: 
                            x_ref, x_dst  = tf.split(x, 2, axis=0)
                            x_dst, gamma, beta = ops_UNIT.apply_ins_norm_2d_like(x_dst, x_ref)
                            new_gamma_list.append(gamma)
                            new_beta_list.append(beta)
                            x = tf.concat([x_ref, x_dst], axis=0)
                        else:
                            raise 'at most one of `fore_half` and `post_half` can be True !'
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
            for i in range(0, self.n_gen_decoder-1) :
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=2, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')
            # x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='deconv_out')

            return x, new_gamma_list, new_beta_list
    # END of DECODERS
    ##############################################################################

    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        
        out_A, gamma_list_A, beta_list_A = self.generator(out, self.is_training, post_half=True, scope="generator_A")
        out_B, gamma_list_B, beta_list_B = self.generator(out, self.is_training, fore_half=True, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B

    def generate_a2b(self, x_A, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out, _, _ = self.generator(out, self.is_training, gamma_list, beta_list, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out, _, _ = self.generator(out, self.is_training, gamma_list, beta_list, reuse=True, scope="generator_A")

        return out, shared

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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



class UNIT_AdaINCycle(UNIT_AdaIN):
    def init_net(self, args):
        assert args.pretrained_path is not None
        if args.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
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
        # if args.pretrained_path is not None:
        #     self.saverPart.restore(self.sess, args.pretrained_path)
        #     print('restored from pretrained_path:', args.pretrained_path)
        # elif self.ckpt_path is not None:
        #     self.saver.restore(self.sess, self.ckpt_path)
        #     print('restored from ckpt_path:', self.ckpt_path)

    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B = self.translation(domain_A, domain_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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

class UNIT_VAEGAN_recon_CombData(UNIT_VAEGAN_recon):
    def _define_input(self):
        ## Input param
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        domain_A = self.domain_A
        domain_B = self.domain_B
        self.domain_A = tf.concat([domain_A, domain_B], axis=0)
        self.domain_B = tf.concat([domain_A, domain_B], axis=0)

class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes(UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes):
    # def init_net(self, args):
    #     assert args.pretrained_path is not None
    #     if args.pretrained_path is not None:
    #         var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A') \
    #             + tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_B')
    #         var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
    #         self.saverPart = tf.train.Saver(var, max_to_keep=5)
    #     if args.pretrained_vgg_path is not None:
    #         var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
    #         var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
    #         self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
    #     if args.test_model_path is not None:
    #         var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
    #         var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
    #         self.saverTest = tf.train.Saver(var, max_to_keep=5)
    #     self.summary_writer = tf.summary.FileWriter(args.model_dir)
    #     sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
    #             summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
    #     if args.phase == 'train':
    #         gpu_options = tf.GPUOptions(allow_growth=True)
    #     else:
    #         gpu_options = tf.GPUOptions(allow_growth=True)
    #     sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    #     self.sess = sv.prepare_or_wait_for_session(config=sess_config)
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

    ##############################################################################
    # BEGIN of DECODERS
    def generator(self, x, is_training=True, gamma_list=None, beta_list=None, fore_half=False, post_half=False, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        # channel = 256
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        # channel_vgg = gamma.get_shape().as_list()[-1]
                        # x = ops_UNIT.conv(x, channel_vgg, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='convIns_in_'+str(i))
                        if fore_half or post_half:
                            assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                            x1, x2 = tf.split(x, 2, axis=0)
                            if fore_half:
                                x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                            if post_half:
                                x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                            x = tf.concat([x1,x2], axis=0)
                        else:
                            ## for generator_a2b and generator_b2a
                            assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                            x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)
                        # x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='convIns_out_'+str(i))


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

    def encoder_spec(self, x, is_training=True, reuse=False, scope="part1"):
        channel = self.ngf
        gamma_list, beta_list = [], []
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
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)

            return gamma_list, beta_list

    def ins_specific_branch(self, x, is_training=False, scope='VAEGAN_recon', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            gamma_list, beta_list = self.encoder_spec(x, is_training, scope="part1")
        return gamma_list, beta_list

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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
        self._define_output()


class UNIT_MultiDecSpecificBranchFromImg_Cycle_ChangeRes(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    ##############################################################################
    # BEGIN of DECODERS
    def generator_spec(self, x, is_training=True, reuse=False, scope="part4"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        gamma_list, beta_list = [], []
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) : 
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)
                if i==(self.n_gen_resblock-1):
                    break
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
            gamma_list = gamma_list[::-1] ## keep the order with the generator
            beta_list = beta_list[::-1] ## keep the order with the generator
            return gamma_list, beta_list
    # END of DECODERS
    ##############################################################################

    def ins_specific_branch(self, x, is_training=False, scope='VAEGAN_recon', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            enc = self.encoder(x, is_training, scope="part1")
            latent = self.share_encoder(enc, is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            gamma_list, beta_list = self.generator_spec(out, is_training, scope="part4")
        return gamma_list, beta_list


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
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

    # def apply_feaMap_mask(self, x, feaMap):
    #     # mask = (tf.sign(feaMap)+1.0)/2.0 ## select the >0 elements as mask
    #     mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
    #     x = x * mask
    #     return x

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
    
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        # feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        # feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        # self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        # self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]
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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

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
            

class UNIT_MultiDecSpecificBranchFromImg_Cycle_ChangeRes_FeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    ##############################################################################
    # BEGIN of DECODERS
    def generator_spec(self, x, is_training=True, reuse=False, scope="part4"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        feaMap_list, gamma_list, beta_list = [], [], []
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) : 
                feaMap_list.append(x)
                mean, var = tf.nn.moments(x, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)
                # if i==(self.n_gen_resblock-1):
                #     break
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
                
            feaMap_list = feaMap_list[::-1] ## keep the order with the generator
            gamma_list = gamma_list[::-1] ## keep the order with the generator
            beta_list = beta_list[::-1] ## keep the order with the generator
            return feaMap_list, gamma_list, beta_list
    # END of DECODERS
    ##############################################################################

    def ins_specific_branch(self, x, is_training=False, scope='VAEGAN_recon', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            enc = self.encoder(x, is_training, scope="part1")
            latent = self.share_encoder(enc, is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            feaMap_list, gamma_list, beta_list = self.generator_spec(out, is_training, scope="part4")
        return feaMap_list, gamma_list, beta_list


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
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

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=False):
        feaMap_list_A_reconA, gamma_list_A_reconA, beta_list_A_reconA = self.ins_specific_branch(x_A, is_training=is_training, scope='VAEGAN_recon_A', reuse=reuse)
        feaMap_list_A_reconB, gamma_list_A_reconB, beta_list_A_reconB = self.ins_specific_branch(x_A, is_training=is_training, scope='VAEGAN_recon_B', reuse=reuse)
        feaMap_list_B_reconA, gamma_list_B_reconA, beta_list_B_reconA = self.ins_specific_branch(x_B, is_training=is_training, scope='VAEGAN_recon_A', reuse=reuse)
        feaMap_list_B_reconB, gamma_list_B_reconB, beta_list_B_reconB = self.ins_specific_branch(x_B, is_training=is_training, scope='VAEGAN_recon_B', reuse=reuse)
        feaMap_list_ab_reconA, gamma_list_ab_reconA, beta_list_ab_reconA = self.ins_specific_branch(x_ab, is_training=is_training, scope='VAEGAN_recon_A', reuse=reuse)
        feaMap_list_ab_reconB, gamma_list_ab_reconB, beta_list_ab_reconB = self.ins_specific_branch(x_ab, is_training=is_training, scope='VAEGAN_recon_B', reuse=reuse)
        feaMap_list_ba_reconA, gamma_list_ba_reconA, beta_list_ba_reconA = self.ins_specific_branch(x_ba, is_training=is_training, scope='VAEGAN_recon_A', reuse=reuse)
        feaMap_list_ba_reconB, gamma_list_ba_reconB, beta_list_ba_reconB = self.ins_specific_branch(x_ba, is_training=is_training, scope='VAEGAN_recon_B', reuse=reuse)

        num = len(feaMap_list_A_reconA)
        ## Use source domain ecnoder to extract the content feature map
        content_loss = 0.
        for i in range(num):
            content_loss += tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_A_reconA[i], feaMap_list_ab_reconA[i])) \
                        + tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_B_reconB[i], feaMap_list_ba_reconB[i]))

        ## Use target domain ecnoder to extract the style stastics, i.e. mean and var
        style_loss = 0.
        for i in range(num):
            style_loss += ops_UNIT.L2_loss(gamma_list_ab_reconB[i], gamma_list_B_reconB[i]) \
                        + ops_UNIT.L2_loss(beta_list_ab_reconB[i], beta_list_B_reconB[i])
            style_loss += ops_UNIT.L2_loss(gamma_list_ba_reconA[i], gamma_list_A_reconA[i]) \
                        + ops_UNIT.L2_loss(beta_list_ba_reconA[i], beta_list_A_reconA[i])

        return content_loss, style_loss

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        _, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        _, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm


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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 3*style_loss + 0.5*content_loss
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss_noGAN(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        _, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        _, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm


        """ Define Loss """
        # G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        # G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        # D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        # D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

        Generator_A_loss = self.L1_weight * l1_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_bab_loss

        Generator_B_loss = self.L1_weight * l1_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab + \
                           self.KL_weight * enc_loss + \
                           self.KL_cycle_weight * enc_aba_loss

        # Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        # Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1.*style_loss + 1.*content_loss
        # self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss


        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        # self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)
        self.D_optim = None

        """" Summary """
        # self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        # self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.lr_sum = tf.summary.scalar("lr", self.lr)
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
        # self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        # self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        # self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.content_loss_sum, self.style_loss_sum])
        # self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss):
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

    def vgg_net(self, x, is_training=False, reuse=False, scope="style_content_vgg"):
        vgg = tensorflow_vgg.vgg19.Vgg19_style('./weights/vgg19.npy')
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


            # fea_list[0] = tf.Print(fea_list[0], [tf.reduce_max(fea_list[0]),tf.reduce_min(fea_list[0])], summarize=40, message="fea_list[0] is:")
            # pdb.set_trace()
        return fea_list
        #     gamma_list, beta_list = [], [] 
        #     for fea in fea_list:
        #         mean, var = tf.nn.moments(fea, [1,2])
        #         gamma = mean - 1.
        #         beta = var
        #         gamma_list.append(gamma)
        #         beta_list.append(beta)
        # return fea_list, gamma_list, beta_list

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE):
        x = tf.concat([x_A, x_B, x_ab, x_ba], axis=0)
        # feaMap_list, gamma_list, beta_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list_A = [tf.split(feaMap, 4, axis=0)[0] for feaMap in feaMap_list]
        feaMap_list_B = [tf.split(feaMap, 4, axis=0)[1] for feaMap in feaMap_list]
        feaMap_list_ab = [tf.split(feaMap, 4, axis=0)[2] for feaMap in feaMap_list]
        feaMap_list_ba = [tf.split(feaMap, 4, axis=0)[3] for feaMap in feaMap_list]

        # gamma_list_A = [tf.split(gamma, 4, axis=0)[0] for gamma in gamma_list]
        # gamma_list_B = [tf.split(gamma, 4, axis=0)[1] for gamma in gamma_list]
        # gamma_list_ab = [tf.split(gamma, 4, axis=0)[2] for gamma in gamma_list]
        # gamma_list_ba = [tf.split(gamma, 4, axis=0)[3] for gamma in gamma_list]
        
        # beta_list_A = [tf.split(beta, 4, axis=0)[0] for beta in beta_list]
        # beta_list_B = [tf.split(beta, 4, axis=0)[1] for beta in beta_list]
        # beta_list_ab = [tf.split(beta, 4, axis=0)[2] for beta in beta_list]
        # beta_list_ba = [tf.split(beta, 4, axis=0)[3] for beta in beta_list]

        num = len(feaMap_list_A)
        ## Use source domain ecnoder to extract the content feature map
        content_loss = 0.
        for i in range(num):
        # for i in range(2,num):
            # content_loss += ((float(i+1)/float(num))**2)*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_A[i], feaMap_list_ab[i])) \
            #             + ((float(i+1)/float(num))**2)*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_B[i], feaMap_list_ba[i]))

            content_loss += (float(i+1)/float(num))*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_A[i], feaMap_list_ab[i])) \
                        + (float(i+1)/float(num))*tf.reduce_mean(ops_UNIT.L1_loss(feaMap_list_B[i], feaMap_list_ba[i]))

        ## Use target domain ecnoder to extract the style stastics, i.e. mean and var
        style_loss = 0.
        for i in range(num):
            style_loss += ops_UNIT.L1_loss(ops_UNIT.gram_matrix(feaMap_list_ab[i]), ops_UNIT.gram_matrix(feaMap_list_B[i]))
            style_loss += ops_UNIT.L1_loss(ops_UNIT.gram_matrix(feaMap_list_ba[i]), ops_UNIT.gram_matrix(feaMap_list_A[i]))

        return content_loss, style_loss

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        _, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        _, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm


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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask, UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_EncStyleContentLoss):
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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1.*style_loss + 1.*content_loss
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask, UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_VggStyleContentLoss):
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

    def vgg_net(self, x, is_training=False, reuse=False, scope="style_content_vgg"):
        vgg = tensorflow_vgg.vgg19.Vgg19_style('./weights/vgg19.npy')
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


            # fea_list[0] = tf.Print(fea_list[0], [tf.reduce_max(fea_list[0]),tf.reduce_min(fea_list[0])], summarize=40, message="fea_list[0] is:")
            # pdb.set_trace()
        return fea_list
        #     gamma_list, beta_list = [], [] 
        #     for fea in fea_list:
        #         mean, var = tf.nn.moments(fea, [1,2])
        #         gamma = mean - 1.
        #         beta = var
        #         gamma_list.append(gamma)
        #         beta_list.append(beta)
        # return fea_list, gamma_list, beta_list

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE):
        x = tf.concat([x_A, x_B, x_ab, x_ba], axis=0)
        # feaMap_list, gamma_list, beta_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list_A = [tf.split(feaMap, 4, axis=0)[0] for feaMap in feaMap_list]
        feaMap_list_B = [tf.split(feaMap, 4, axis=0)[1] for feaMap in feaMap_list]
        feaMap_list_ab = [tf.split(feaMap, 4, axis=0)[2] for feaMap in feaMap_list]
        feaMap_list_ba = [tf.split(feaMap, 4, axis=0)[3] for feaMap in feaMap_list]

        # gamma_list_A = [tf.split(gamma, 4, axis=0)[0] for gamma in gamma_list]
        # gamma_list_B = [tf.split(gamma, 4, axis=0)[1] for gamma in gamma_list]
        # gamma_list_ab = [tf.split(gamma, 4, axis=0)[2] for gamma in gamma_list]
        # gamma_list_ba = [tf.split(gamma, 4, axis=0)[3] for gamma in gamma_list]
        
        # beta_list_A = [tf.split(beta, 4, axis=0)[0] for beta in beta_list]
        # beta_list_B = [tf.split(beta, 4, axis=0)[1] for beta in beta_list]
        # beta_list_ab = [tf.split(beta, 4, axis=0)[2] for beta in beta_list]
        # beta_list_ba = [tf.split(beta, 4, axis=0)[3] for beta in beta_list]

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

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 5e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 2e3*style_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 5e3*style_loss + 1e1*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskE2E_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def init_net(self, args):
        # assert args.pretrained_path is not None
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
            latent = self.share_encoder(feaMap_list[-1], is_training, scope="part2")
            out = self.share_generator(latent, is_training, scope="part3")
            out = self.generator(out, is_training, scope="part4")
        return feaMap_list, gamma_list, beta_list, out, latent

    def _build_model(self):
        self._define_input()        
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A, specA, latent_specA = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B, specB, latent_specB = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
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
                fake_specA_logit = self.discriminator(self.fake_A_pool.query(specA), reuse=True, scope="discriminator_A")
                fake_specB_logit = self.discriminator(self.fake_A_pool.query(specB), reuse=True, scope="discriminator_B")
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")                
                fake_specA_logit = self.discriminator(specA, reuse=True, scope="discriminator_A")
                fake_specB_logit = self.discriminator(specB, reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan) \
                    # + ops_UNIT.generator_loss(fake_specA_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan) \
                    # + ops_UNIT.generator_loss(fake_specB_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        enc_loss = ops_UNIT.KL_divergence(shared) \
                    + ops_UNIT.KL_divergence(latent_specA) \
                    + ops_UNIT.KL_divergence(latent_specB)
        enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) \
                    + ops_UNIT.L1_loss(specA, domain_A)
        l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) \
                    + ops_UNIT.L1_loss(specB, domain_B)
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

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 0e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
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

class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_EncIN(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
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
                             is_training=is_training, norm_fn='instance', scope='resblock_'+str(i))

            return x
    # END of ENCODERS
    ##############################################################################

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

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
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
                    # x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    # x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
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
                        # x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)

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

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 0e1*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e3*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e1*content_loss
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


## ref pip install guided-filter-tf
## https://github.com/wuhuikai/DeepGuidedFilter
from guided_filter_tf.guided_filter import fast_guided_filter
class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss_GFhigh(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)

    def _build_model(self):
        self._define_input()
        domain_A_h4 = self.domain_A 
        domain_B_h4 = self.domain_B
        domain_A_h2 = tf.image.resize_nearest_neighbor(domain_A_h4, [int(0.5*self.img_h), int(0.5*self.img_w)])
        domain_B_h2 = tf.image.resize_nearest_neighbor(domain_B_h4, [int(0.5*self.img_h), int(0.5*self.img_w)])
        domain_A = tf.image.resize_nearest_neighbor(domain_A_h4, [int(0.25*self.img_h), int(0.25*self.img_w)])
        domain_B = tf.image.resize_nearest_neighbor(domain_B_h4, [int(0.25*self.img_h), int(0.25*self.img_w)])

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

            self.x_ab_h2 = self._guided_filter(domain_A, x_ab, domain_A_h2, scope='guided_filter_A')
            self.x_ba_h2 = self._guided_filter(domain_B, x_ba, domain_B_h2, scope='guided_filter_B')
            self.x_ab_h4 = self._guided_filter(domain_A, x_ab, domain_A_h4, scope='guided_filter_A', reuse=True)
            self.x_ba_h4 = self._guided_filter(domain_B, x_ba, domain_B_h4, scope='guided_filter_B', reuse=True)

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

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 5e2*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
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

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, x_ab_h2, x_ba_h2, x_ab_h4, x_ba_h4, feaMapA, feaMapB = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.x_ab_h2, self.x_ba_h2, self.x_ab_h4, self.x_ba_h4, self.feaMask_list_A[0], self.feaMask_list_B[0]], \
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
        x_ba_h2_img = unprocess_image(x_ba_h2, 127.5, 127.5)
        x_ab_h2_img = unprocess_image(x_ab_h2, 127.5, 127.5)
        x_ba_h4_img = unprocess_image(x_ba_h4, 127.5, 127.5)
        x_ab_h4_img = unprocess_image(x_ab_h4, 127.5, 127.5)
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
        save_images(x_ba_h2_img, [self.batch_size, 1],
                    '{}/x_ba_h2_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_h2_img, [self.batch_size, 1],
                    '{}/x_ab_h2_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_h4_img, [self.batch_size, 1],
                    '{}/x_ba_h4_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_h4_img, [self.batch_size, 1],
                    '{}/x_ab_h4_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ShareSB(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def init_net(self, args):
        assert args.pretrained_path is not None
        if args.pretrained_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A')
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

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)
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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_ConvFeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def apply_feaMap_mask(self, x, feaMap, scope='conv_feaMask', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('UNIT', reuse=reuse):
            _, _, _, c = feaMap.get_shape().as_list()
            # feaMap = ops_UNIT.activation(x, activation_fn='leaky')
            feaMap = ops_UNIT.conv(feaMap, feaMap.get_shape().as_list()[-1], kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None)
            mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
            mask = mask/2. + 1./2. ## norm [0.5, 1] to [0.75, 1]
            mask = tf.image.resize_images(mask, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            x = x * mask
            return x

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_vgg_path is not None:
            self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
            print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)
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
                # for g_iter in range(2):
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_BlurFeaMask(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def apply_feaMap_mask(self, x, feaMap):
        # mask = (tf.sign(feaMap)+1.0)/2.0 ## select the >0 elements as mask
        mask = tf.sigmoid(feaMap) ## select the >0 elements as mask
        # mask = mask/2. - 1 ## norm [0.5, 1] to [0, 1]
        mask = mask/2. + 1./2. ## norm [0.5, 1] to [0.75, 1]
        # mask = mask/5.*2. + 0.6 ## norm [0.5, 1] to [0.8, 1]
        # mask = mask/4. + 3./4. ## norm [0.5, 1] to [0.875, 1]
        # mask = tf.ones_like(mask) ## norm [0.5, 1] to [1, 1]
        mask = tf.image.resize_images(mask, x.get_shape().as_list()[1:3], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        mask = ops_UNIT.conv_gaussian_blur(mask, filter_size=3, sigma=3, name='gaussian_blur', padding='SAME')
        x = x * mask
        return x


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolShare(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    def share_encoder_generator(self, x, is_training=True, reuse=False, scope="share_encoder_generator"):
        channel = self.ngf * pow(2, self.n_encoder-1)
        with tf.variable_scope(scope, reuse=reuse) :
            enc_list = []
            argmax_list = []
            # shape_list = [tf.shape(x)]
            for i in range(0, self.n_enc_share) :
                enc_list.append(x)
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='enc_resblock_'+str(i))
                if i < self.n_enc_share-1:
                    x, x_argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_'+str(i))
                    argmax_list.append(x_argmax)
                    # shape_list.append(tf.shape(x))

            x = ops_UNIT.gaussian_noise_layer(x)
            latent = x

            for i in range(0, self.n_gen_share) :
                if i < self.n_enc_share-1:
                    # pdb.set_trace()
                    # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                    x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='gen_resblock_'+str(i))


            return latent, x
    # END of SHARED LAYERS
    ##############################################################################

    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)
        shared, out = self.share_encoder_generator(out, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        shared, out = self.share_encoder_generator(out, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        shared, out = self.share_encoder_generator(out, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out, shared


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_UnpoolEncGen(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    ##############################################################################
    # BEGIN of ENCODERS
    def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
        channel = self.ngf
        argmax_list = []
        with tf.variable_scope(scope, reuse=reuse) :
            x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

            for i in range(1, self.n_encoder) :
                x = ops_UNIT.conv(x, channel*2, kernel=3, stride=1, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
                x, x_argmax = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_'+str(i))
                argmax_list.append(x_argmax)
                channel *= 2

            # channel = 256
            for i in range(0, self.n_enc_resblock) :
                x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                             normal_weight_init=self.normal_weight_init,
                             is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

            return x, argmax_list
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
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
                # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    def generator_one(self, x, argmax_list, feaMap_list=None, gamma_list=None, beta_list=None, is_training=True, reuse=False, scope="generator"):
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
                # x = ops_UNIT.unpool_layer2x2_batch(x, argmax_list[-1-i])
                x = ops_UNIT.unpool_2d(x, argmax_list[-1-i])
                x = ops_UNIT.deconv(x, channel//2, kernel=3, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='deconv_'+str(i))
                channel = channel // 2

            x = ops_UNIT.deconv(x, self.output_c_dim, kernel=1, stride=1, normal_weight_init=self.normal_weight_init, activation_fn='tanh', scope='deconv_tanh')

            return x
    # END of DECODERS
    ##############################################################################

    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        enc_A, argmax_list_A = self.encoder(x_A, self.is_training, scope="encoder_A")
        enc_B, argmax_list_B = self.encoder(x_B, self.is_training, scope="encoder_B")
        out = tf.concat([enc_A, enc_B], axis=0)
        argmax_list = [tf.concat([argmax_list_A[i], argmax_list_B[i]], axis=0) for i in range(len(argmax_list_A))]
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, argmax_list, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out, argmax_list = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, argmax_list, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out, argmax_list = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        shared = self.share_encoder(out, self.is_training, reuse=True)
        out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, argmax_list, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out, shared


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Pair(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

            # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
            real_A_logit = self.discriminator(tf.concat([domain_B, domain_A], axis=-1), scope="discriminator_A")
            real_B_logit = self.discriminator(tf.concat([domain_A, domain_B], axis=-1), scope="discriminator_B")

            if self.replay_memory :
                self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
                self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
                # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
                fake_A_logit = self.discriminator(self.fake_A_pool.query(tf.concat([domain_B, x_ba], axis=-1)), reuse=True, scope="discriminator_A") # replay memory
                fake_B_logit = self.discriminator(self.fake_B_pool.query(tf.concat([domain_A, x_ab], axis=-1)), reuse=True, scope="discriminator_B") # replay memory
            else :
                # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
                fake_A_logit = self.discriminator(tf.concat([domain_B, x_ba], axis=-1), reuse=True, scope="discriminator_A")
                fake_B_logit = self.discriminator(tf.concat([domain_A, x_ab], axis=-1), reuse=True, scope="discriminator_B")

            """ Generated Image """
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_Triplet(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, gamma_list_A, beta_list_A, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, gamma_list_A, beta_list_A)
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
            self.fake_B, _ = self.generate_a2b(domain_A) # for test without applying Instance Norm
            self.fake_A, _ = self.generate_b2a(domain_B) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(tf.concat([real_A_logit,real_A_logit],axis=0), tf.concat([real_B_logit,fake_A_logit],axis=0), smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(tf.concat([real_B_logit,real_B_logit],axis=0), tf.concat([real_A_logit,fake_B_logit],axis=0), smoothing=self.smoothing, use_lasgan=self.use_lsgan)

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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()




class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_LearnIN(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
    
    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=1:
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
                if i<=1:
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

    ## TODO
    def learn_gamma_beta_feaMask(self, feaMap_list, gamma_list, beta_list, reuse=False, scope='leanrnIN'):
        with tf.variable_scope(scope, reuse=reuse) :
            return feaMap_list, gamma_list, beta_list

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        with tf.variable_scope('UNIT'):
            feaMap_list_A, gamma_list_A, beta_list_A = self.learn_gamma_beta_feaMask(self, feaMap_list_A, gamma_list_A, beta_list_A, scope='leanrnIN_A')
            feaMap_list_B, gamma_list_B, beta_list_B = self.learn_gamma_beta_feaMask(self, feaMap_list_B, gamma_list_B, beta_list_B, scope='leanrnIN_B')
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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 0.*style_loss + 0.*content_loss
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


## ref pip install guided-filter-tf
## https://github.com/wuhuikai/DeepGuidedFilter
from guided_filter_tf.guided_filter import fast_guided_filter
class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb
            # r=3
            # eps=1e-8
            x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A', reuse=True)
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B', reuse=True)
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, \
            x_aa, x_ba, x_ab, x_bb, feaMapA, feaMapB, x_aa_GF, x_ba_GF, x_ab_GF, x_bb_GF = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.feaMask_list_A[0], self.feaMask_list_B[0], \
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF], \
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
        x_aa_GF_img = unprocess_image(x_aa_GF, 127.5, 127.5)
        x_ba_GF_img = unprocess_image(x_ba_GF, 127.5, 127.5)
        x_ab_GF_img = unprocess_image(x_ab_GF, 127.5, 127.5)
        x_bb_GF_img = unprocess_image(x_bb_GF, 127.5, 127.5)
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
        save_images(x_aa_GF_img, [self.batch_size, 1],
                    '{}/x_aa_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_ba_GF_img, [self.batch_size, 1],
                    '{}/x_ba_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
        save_images(x_ab_GF_img, [self.batch_size, 1],
                    '{}/x_ab_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        save_images(x_bb_GF_img, [self.batch_size, 1],
                    '{}/x_bb_GF_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
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
            

class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ConvGF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF):
    def _guided_map(self, x, is_training, activation_fn=ops_UNIT.LeakyReLU):
        channel_num = self.ngf
        x = slim.conv2d(x, channel_num, 1, 1, activation_fn=None)
        x = ops_UNIT.adaptive_BN(x, is_training)
        # x = ops_UNIT.adaptive_IN(x)
        x = activation_fn(x)
        x = slim.conv2d(x, 3, 1, 1, activation_fn=None)
        return x

    def _guided_filter(self, lr_x, lr_y, hr_x, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        with tf.variable_scope(scope, reuse=reuse) :
            x_in = tf.concat([lr_x, hr_x], axis=0)
            x_out = self._guided_map(x_in, self.is_training)
            lr_x, hr_x = tf.split(x_out, 2, axis=0)
            return fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_GF):
    def init_net(self, args):
        if args.test_model_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverTest = tf.train.Saver(var, max_to_keep=5)
        if args.pretrained_common_path is not None:
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
            var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
            self.saverCommon = tf.train.Saver(var, max_to_keep=5)
        self.summary_writer = tf.summary.FileWriter(args.model_dir)
        sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
                summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
        if args.phase == 'train':
            gpu_options = tf.GPUOptions(allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def _dynamic_filters(self, x, is_training, filter_size=9, activation_fn=ops_UNIT.LeakyReLU, use_dynamic_bias=False):
        channel_num = self.ngf
        x = ops_UNIT.conv(x, channel_num, kernel=3, stride=1, pad=1, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_in')
        for i in range(0, 3) :
            x = ops_UNIT.resblock(x, channel_num, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
                         normal_weight_init=self.normal_weight_init,
                         is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))
        # dynamic_filters = ops_UNIT.conv(x, filter_size**2, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_filter')
        # dynamic_filters = tf.nn.softmax(dynamic_filters, axis=-1)

        kernel_1 = tf.Variable(tf.zeros([1, 1, channel_num, filter_size**2]))
        conv_kernel_1 = tf.nn.conv2d(x, kernel_1, [1,1,1,1], padding='SAME')
        dense = tf.sparse_to_dense(sparse_indices=[int((filter_size**2)/2)], output_shape=[filter_size**2], sparse_values=[1.0])
        biases_1 = tf.Variable(dense)
        dynamic_filters = tf.nn.bias_add(conv_kernel_1, biases_1)

        if use_dynamic_bias:
            dynamic_bias = ops_UNIT.conv(x, 1, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn=None, scope='conv_bias')
        else:
            dynamic_bias = 0

        return dynamic_filters, dynamic_bias

    def _guided_filter(self, x, G, hr_x=None, r=3, eps=1e-8, nhwc=True, scope='guided_filter', reuse=None):
        with tf.variable_scope(scope, reuse=reuse) :
            filter_size = 5
            pad = int((filter_size-1)/2)
            x_out = tf.zeros_like(G)
            B, H, W, C = x_out.get_shape().as_list()
            dynamic_filters, dynamic_bias = self._dynamic_filters(x, self.is_training, filter_size)
            G = tf.pad(G+dynamic_bias, [[0,0], [pad, pad], [pad, pad], [0,0]], "REFLECT")
            # convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding='VALID')
            # patches = tf.reshape(tf.extract_image_patches(x, [1, filter_size, filter_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"), [B,H,W,C,-1])
            # filter_patches = tf.tile(tf.expand_dims(dynamic_filters, axis=-2), [1,1,1,C,1])
            # x_out = tf.reduce_sum(filter_patches*patches, axis=[-1], keepdims=False)
            patches = tf.reshape(tf.extract_image_patches(G, [1, filter_size, filter_size, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding="VALID"), [B,H,W,-1,C])
            filter_patches = tf.tile(tf.expand_dims(dynamic_filters, axis=-1), [1,1,1,1,C])
            x_out = tf.reduce_sum(filter_patches*patches, axis=[-2], keepdims=False)
            return x_out

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('GuidedFilter'):
            # r=3
            # eps=1e-8
            x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A', reuse=True)
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B', reuse=True)
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('UNIT'):
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
        # G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        G_vars = [var for var in t_vars if ('GuidedFilter' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def train(self, args):
        """Train SG-GAN"""
        self.sess.run(self.init_op)
        self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
        if args.pretrained_common_path is not None:
            self.saverCommon.restore(self.sess, args.pretrained_common_path)
            print('restored from pretrained_common_path:', args.pretrained_common_path)
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


class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF_OnlyCrossDomain(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_ImgConvDynGF):

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i])/2.+1./2 for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i])/2.+1./2 for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
            self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('GuidedFilter'):
            # r=3
            # eps=1e-8
            # x_aa = self._guided_filter(domain_A, x_aa, domain_A, scope='guided_filter_A')
            x_ab = self._guided_filter(domain_A, x_ab, domain_A, scope='guided_filter_A')
            x_aba = self._guided_filter(domain_A, x_aba, domain_A, scope='guided_filter_A', reuse=True)
            # x_bb = self._guided_filter(domain_B, x_bb, domain_B, scope='guided_filter_B')
            x_ba = self._guided_filter(domain_B, x_ba, domain_B, scope='guided_filter_B')
            x_bab = self._guided_filter(domain_B, x_bab, domain_B, scope='guided_filter_B', reuse=True)
            self.x_aa_GF, self.x_ba_GF, self.x_ab_GF, self.x_bb_GF = x_aa, x_ba, x_ab, x_bb

        with tf.variable_scope('UNIT'):
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
        # G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
        G_vars = [var for var in t_vars if ('GuidedFilter' in var.name)]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # pdb.set_trace()
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

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()



class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_Grad(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def gradient_map(self, x):
        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        _, _, _, ch = x.get_shape().as_list()
        kernels = []
        kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), ch) )
        kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), ch) )
        kernel = tf.constant(np.stack(kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        gradient_map = tf.abs(tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding="VALID"))
        # pdb.set_trace()
        return gradient_map

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
            self.grad_A_map = self.gradient_map(domain_A)
            self.grad_B_map = self.gradient_map(domain_B)
            self.grad_ab_map = self.gradient_map(self.x_ab)
            self.grad_ba_map = self.gradient_map(self.x_ba)

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

        grad_loss_a = tf.reduce_mean(ops_UNIT.L2_loss(self.grad_ab_map, self.grad_A_map))
        grad_loss_b = tf.reduce_mean(ops_UNIT.L2_loss(self.grad_ba_map, self.grad_B_map))
        self.grad_loss = grad_loss_a + grad_loss_b

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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 500.0*self.grad_loss
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
        self.grad_loss_a_sum = tf.summary.scalar("grad_loss_a", grad_loss_a)
        self.grad_loss_b_sum = tf.summary.scalar("grad_loss_b", grad_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, self.grad_loss_a_sum, self.grad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()

    def sample_model(self, sample_dir, epoch, idx):
        img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, grad_A_map, grad_B_map, grad_ab_map, grad_ba_map = self.sess.run(
            [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
            self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
            self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.grad_A_map, self.grad_B_map, self.grad_ab_map, self.grad_ba_map], \
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
        grad_A_map_img = unprocess_image((grad_A_map/3.-0.5)*2, 127.5, 127.5) ## gradient value is in about [0,3], norm to [-1,1]
        grad_B_map_img = unprocess_image((grad_B_map/3.-0.5)*2, 127.5, 127.5)
        grad_ab_map_img = unprocess_image((grad_ab_map/3.-0.5)*2, 127.5, 127.5)
        grad_ba_map_img = unprocess_image((grad_ba_map/3.-0.5)*2, 127.5, 127.5)
        grad_A_map_img = np.tile(np.mean(grad_A_map_img, keepdims=True), [1,1,1,3])
        grad_B_map_img = np.tile(np.mean(grad_B_map_img, keepdims=True), [1,1,1,3])
        grad_ab_map_img = np.tile(np.mean(grad_ab_map_img, keepdims=True), [1,1,1,3])
        grad_ba_map_img = np.tile(np.mean(grad_ba_map_img, keepdims=True), [1,1,1,3])
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
        try:
            # pdb.set_trace()
            save_images(grad_A_map_img, [self.batch_size, 1],
                        '{}/grad_A_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_B_map_img, [self.batch_size, 1],
                        '{}/grad_B_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_ab_map_img, [self.batch_size, 1],
                        '{}/grad_ab_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
            save_images(grad_ba_map_img, [self.batch_size, 1],
                        '{}/grad_ba_map_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
        except:
            print("Oops!",sys.exc_info()[0],"occured.")

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
            


class UNIT_MultiVggSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask):
    def init_net(self, args):
        # assert args.pretrained_vgg_path is not None
        # if args.pretrained_vgg_path is not None:
        #     var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
        #     var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
        #     self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
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

    # def vgg_net(self, x, is_training=False, reuse=False):
    #     _, end_points = nets.vgg.vgg_19_style_full(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
    #     # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
    #     # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
    #     # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
    #     # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
    #     layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
    #     # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
    #     fea_list, gamma_list, beta_list = [], [], []
    #     # for i in [1, 2, 3]:
    #     # for i in [1, 2, 3, 4]:
    #     for i in [1, 2, 3, 4, 5]:
    #     # for i in [3]: 
    #         fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
    #         mean, var = tf.nn.moments(fea, [1,2])
    #         gamma = mean - 1.
    #         beta = var
    #         fea_list.append(fea)
    #         gamma_list.append(gamma)
    #         beta_list.append(beta)
        
    #     return fea_list, gamma_list, beta_list

    def vgg_net_tf_official(self, x, is_training=False, reuse=False):
        assert self.pretrained_vgg_path is not None
        def vgg_preprocess(x): ## from [-1,1] back to [0,255], then subtracts vgg mean
            VGG_MEAN = [123.68, 116.779, 103.939]     # This is R-G-B for Imagenet
            x = unprocess_image(x, 127.5, 127.5)
            x_R, x_G, x_B = tf.split(x, 3, axis=-1)
            x_R -= VGG_MEAN[0]
            x_G -= VGG_MEAN[1]
            x_B -= VGG_MEAN[2]
            x = tf.concat([x_R, x_G, x_B], axis=-1)
            # x = tf.concat([x_B, x_G, x_R], axis=-1)
            return x

        x = vgg_preprocess(x) 
        _, end_points = nets.vgg.vgg_19_style_full(x, is_training=is_training, spatial_squeeze=False, reuse=reuse)
        # vgg_conv1_1 = end_points['vgg_19/conv1/conv1_1'] ## 64 channels
        # vgg_conv2_1 = end_points['vgg_19/conv2/conv2_1'] ## 128 channels
        # vgg_conv3_1 = end_points['vgg_19/conv3/conv3_1'] ## 256 channels
        # vgg_conv4_1 = end_points['vgg_19/conv4/conv4_1'] ## 512 channels
        layer_dic = {1:1, 2:1, 3:1, 4:1, 5:1} # block:layer
        # layer_dic = {1:2, 2:2, 3:1, 4:4, 5:4}
        fea_list, gamma_list, beta_list = [], [], []
        # for i in [1, 2, 3]:
        # for i in [1, 2, 3, 4]:
        for i in [1, 2, 3, 4, 5]:
        # for i in [3]: 
            fea = end_points['vgg_19/conv%d/conv%d_%d'%(i,i,layer_dic[i])]
            mean, var = tf.nn.moments(fea, [1,2])
            gamma = mean - 1.
            beta = var
            fea_list.append(fea)
            gamma_list.append(gamma)
            beta_list.append(beta)
        
        return fea_list, gamma_list, beta_list

    def vgg_net(self, x, is_training=False, reuse=False, scope="style_content_vgg"):
        vgg = tensorflow_vgg.vgg19.Vgg19_style('./weights/vgg19.npy')
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


            # fea_list[0] = tf.Print(fea_list[0], [tf.reduce_max(fea_list[0]),tf.reduce_min(fea_list[0])], summarize=40, message="fea_list[0] is:")
            # pdb.set_trace()
            gamma_list, beta_list = [], [] 
            for fea in fea_list:
                mean, var = tf.nn.moments(fea, [1,2])
                gamma = mean - 1.
                beta = var
                gamma_list.append(gamma)
                beta_list.append(beta)
        return fea_list, gamma_list, beta_list

    def encoder_style_content_loss(self, x_A, x_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE):
        x = tf.concat([x_A, x_B, x_ab, x_ba], axis=0)
        feaMap_list, gamma_list, beta_list = self.vgg_net(x, is_training=is_training, reuse=reuse)
        feaMap_list_A = [tf.split(feaMap, 4, axis=0)[0] for feaMap in feaMap_list]
        feaMap_list_B = [tf.split(feaMap, 4, axis=0)[1] for feaMap in feaMap_list]
        feaMap_list_ab = [tf.split(feaMap, 4, axis=0)[2] for feaMap in feaMap_list]
        feaMap_list_ba = [tf.split(feaMap, 4, axis=0)[3] for feaMap in feaMap_list]

        gamma_list_A = [tf.split(gamma, 4, axis=0)[0] for gamma in gamma_list]
        gamma_list_B = [tf.split(gamma, 4, axis=0)[1] for gamma in gamma_list]
        gamma_list_ab = [tf.split(gamma, 4, axis=0)[2] for gamma in gamma_list]
        gamma_list_ba = [tf.split(gamma, 4, axis=0)[3] for gamma in gamma_list]
        
        beta_list_A = [tf.split(beta, 4, axis=0)[0] for beta in beta_list]
        beta_list_B = [tf.split(beta, 4, axis=0)[1] for beta in beta_list]
        beta_list_ab = [tf.split(beta, 4, axis=0)[2] for beta in beta_list]
        beta_list_ba = [tf.split(beta, 4, axis=0)[3] for beta in beta_list]

        num = len(feaMap_list_A)
        ## Use source domain ecnoder to extract the content feature map
        content_loss = 0.
        for i in range(num):
            content_loss += tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_A[i], feaMap_list_ab[i])) \
                        + tf.reduce_mean(ops_UNIT.L2_loss(feaMap_list_B[i], feaMap_list_ba[i]))

        ## Use target domain ecnoder to extract the style stastics, i.e. mean and var
        style_loss = 0.
        for i in range(num):
            style_loss += ops_UNIT.L2_loss(gamma_list_ab[i], gamma_list_B[i]) \
                        + ops_UNIT.L2_loss(beta_list_ab[i], beta_list_B[i])
            style_loss += ops_UNIT.L2_loss(gamma_list_ba[i], gamma_list_A[i]) \
                        + ops_UNIT.L2_loss(beta_list_ba[i], beta_list_A[i])

        # content_loss = tf.Print(content_loss, [content_loss], summarize=40, message="content_loss is:")
        # style_loss = tf.Print(style_loss, [style_loss], summarize=40, message="style_loss is:")

        return content_loss, style_loss

    ##############################################################################
    # BEGIN of DECODERS
    def generator_two(self, x, feaMap_list_A, feaMap_list_B, gamma_list, beta_list, is_training=True, reuse=False, scope="generator"):
        channel_original = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            # channel = self.channel_list[-1]
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    channel = self.channel_list[-1-i]
                    if x.get_shape().as_list()[-1]!= channel:
                        x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                    feaMapA, feaMapB, gamma, beta = feaMap_list_A[-1-i], feaMap_list_B[-1-i], gamma_list[-1-i], beta_list[-1-i]
                    assert x.get_shape().as_list()[0]==2*gamma.get_shape().as_list()[0]
                    x1, x2 = tf.split(x, 2, axis=0)
                    ## (x1,x2) is (x_Aa,x_Ba) or (x_Ab,x_Bb)
                    x1 = self.apply_feaMap_mask(x1, feaMapA)
                    x1 = ops_UNIT.apply_ins_norm_2d(x1, gamma, beta)
                    x2 = self.apply_feaMap_mask(x2, feaMapB)
                    x2 = ops_UNIT.apply_ins_norm_2d(x2, gamma, beta)
                    x = tf.concat([x1,x2], axis=0)
                elif channel_original!=x.get_shape().as_list()[-1]:
                    channel = channel_original
                    x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                      
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
        channel_original = self.ngf * pow(2, self.n_encoder - 1)
        with tf.variable_scope(scope, reuse=reuse) :
            # channel = self.channel_list[-1]
            # x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans')
            for i in range(0, self.n_gen_resblock) :
                # if i<=2:
                if i<=0:
                    channel = self.channel_list[-1-i]
                    if x.get_shape().as_list()[-1]!= channel:
                        x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                    
                    if feaMap_list is not None:
                        feaMap = feaMap_list[-1-i]
                        assert x.get_shape().as_list()[0]==feaMap.get_shape().as_list()[0]
                        x = self.apply_feaMap_mask(x, feaMap)

                    if gamma_list is not None and beta_list is not None:
                        gamma, beta = gamma_list[-1-i], beta_list[-1-i]
                        assert x.get_shape().as_list()[0]==gamma.get_shape().as_list()[0]
                        x = ops_UNIT.apply_ins_norm_2d(x, gamma, beta)
                elif channel_original!=x.get_shape().as_list()[-1]:
                    channel = channel_original
                    x = ops_UNIT.conv(x, channel, kernel=1, stride=1, pad=0, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_trans_'+str(i))
                                    
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
    
    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.vgg_net(domain_A, is_training=False, reuse=False)
        feaMap_list_B, gamma_list_B, beta_list_B = self.vgg_net(domain_B, is_training=False, reuse=True)
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
        self.channel_list = [feaMap_list_A[i].get_shape().as_list()[-1] for i in range(len(feaMap_list_A))]
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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e7*style_loss + 1e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss
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
        # self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        # self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
                        self.enc_bab_loss_sum, self.enc_aba_loss_sum, ])
                        # self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()



class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMaskFromOther_EncStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
    def translation(self, x_A, x_B, feaMap_list_A, feaMap_list_A_other, gamma_list_A, beta_list_A, feaMap_list_B, feaMap_list_B_other, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        shared = self.share_encoder(out, self.is_training)
        out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B_other, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A_other, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def _build_model(self):
        self._define_input()
        domain_A = self.domain_A
        domain_B = self.domain_B

        feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
        feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
        feaMap_list_A_other, _, _ = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        feaMap_list_B_other, _, _ = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
        self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
            x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, feaMap_list_A_other, gamma_list_A, beta_list_A, feaMap_list_B, feaMap_list_B_other, gamma_list_B, beta_list_B)
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

        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1.*style_loss + 0.2*content_loss
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



class UNIT_VAEGAN_LatentTransform(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes):
    def Latent_Transform(self, x, is_training=True, reuse=False, scope="Latent_Transform", activation_fn=ops_UNIT.LeakyReLU):
        _, out_h, out_w, out_c = x.get_shape().as_list()

        with tf.variable_scope(scope, reuse=reuse) :
            channel_num = out_c
            channel_step = self.ngf
            _, h, w, _ = x.get_shape().as_list()
            repeat_num = int(np.log2(h/4))

            # Encoder
            encoder_layer_list = []
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)

            for idx in range(repeat_num):
                res = x
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                encoder_layer_list.append(x)
                if idx < repeat_num - 1:
                    # channel_num *= 2
                    channel_num += channel_step
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn)

            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [x_shape[0], np.prod(x_shape[1:])])
            z = x = slim.fully_connected(x, 128, activation_fn=None)

            ## Decoder
            x = slim.fully_connected(z, x_shape[1]*x_shape[2]*channel_num, activation_fn=None)
            x = tf.reshape(x, x_shape)
            
            for idx in range(repeat_num):
                # pdb.set_trace()
                x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
                res = x
                # channel_num = hidden_num * (repeat_num-idx)
                channel_num = x.get_shape().as_list()[-1]
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                if idx < repeat_num - 1:
                    _, h, w, _ = x.get_shape().as_list()
                    x = tf.image.resize_nearest_neighbor(x, [2*h, 2*w])
                    # channel_num -= channel_step
                    channel_num /= 2
                    x = slim.conv2d(x, channel_num, 1, 1, activation_fn=activation_fn)

            out = slim.conv2d(x, out_c, 3, 1, activation_fn=None)

            return out

    def VAEGAN_recon(self, x, latent=None, is_training=True, scope='VAEGAN_recon', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if latent is None:
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

        self.x_aa, latent_A = self.VAEGAN_recon(domain_A, is_training=False, scope='VAEGAN_recon_A')
        self.x_bb, latent_B = self.VAEGAN_recon(domain_B, is_training=False, scope='VAEGAN_recon_B')
        # pdb.set_trace() 
        latent_ab = self.Latent_Transform(latent_A, self.is_training, scope="Latent_Transform_A2B")
        latent_ba = self.Latent_Transform(latent_B, self.is_training, scope="Latent_Transform_B2A")
        self.x_ab, _ = self.VAEGAN_recon(None, latent_ab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.x_ba, _ = self.VAEGAN_recon(None, latent_ba, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        self.fake_B, _ = self.VAEGAN_recon(None, latent_A, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.fake_A, _ = self.VAEGAN_recon(None, latent_B, is_training=False, scope='VAEGAN_recon_A', reuse=True)


        _, latent_m = self.VAEGAN_recon(self.x_ab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        _, latent_n = self.VAEGAN_recon(self.x_ba, is_training=False, scope='VAEGAN_recon_A', reuse=True)
        # pdb.set_trace() 
        latent_bab = self.Latent_Transform(latent_n, self.is_training, scope="Latent_Transform_A2B", reuse=True)
        latent_aba = self.Latent_Transform(latent_m, self.is_training, scope="Latent_Transform_B2A", reuse=True)
        self.x_bab, _ = self.VAEGAN_recon(None, latent_bab, is_training=False, scope='VAEGAN_recon_B', reuse=True)
        self.x_aba, _ = self.VAEGAN_recon(None, latent_aba, is_training=False, scope='VAEGAN_recon_A', reuse=True)

        real_latent_A_logit = self.discriminator(latent_A, scope="discriminator_latent_A")
        real_latent_B_logit = self.discriminator(latent_B, scope="discriminator_latent_B")
        fake_latent_A_logit = self.discriminator(latent_ba, reuse=True, scope="discriminator_latent_A")
        fake_latent_B_logit = self.discriminator(latent_ab, reuse=True, scope="discriminator_latent_B")

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_latent_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_latent_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_latent_A_logit, fake_latent_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_latent_B_logit, fake_latent_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss_a = ops_UNIT.KL_divergence(latent_A)
        # enc_loss_b = ops_UNIT.KL_divergence(latent_B)

        # l1_loss_a = ops_UNIT.L1_loss(self.x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(self.x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(self.x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(self.x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba
                           # self.L1_weight * l1_loss_a + \
                           # self.KL_weight * enc_loss_a

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab
                           # self.L1_weight * l1_loss_b + \
                           # self.KL_weight * enc_loss_b

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
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        # self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
        # self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
        # self.enc_loss_a_sum = tf.summary.scalar("KL_enc_loss_a", enc_loss_a)
        # self.enc_loss_b_sum = tf.summary.scalar("KL_enc_loss_b", enc_loss_b)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()




class UNIT_CycleGAN(UNIT):
    def translation(self, x_A, x_B):
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)

        out_A = self.generator(out, self.is_training, scope="generator_A")
        out_B = self.generator(out, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb #, shared

    def generate_a2b(self, x_A):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_B")

        return out #, shared

    def generate_b2a(self, x_B):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator(out, self.is_training, reuse=True, scope="generator_A")

        return out #, shared

    def _build_model(self):
        self._define_input()
        # self.is_training = tf.placeholder(tf.bool)
        domain_A = self.domain_A
        domain_B = self.domain_B

        with tf.variable_scope('UNIT'):
            """ Define Encoder, Generator, Discriminator """
            x_aa, x_ba, x_ab, x_bb = self.translation(domain_A, domain_B)
            x_bab = self.generate_a2b(x_ba)
            x_aba = self.generate_b2a(x_ab)
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
            self.fake_B = self.generate_a2b(domain_A) # for test
            self.fake_A = self.generate_b2a(domain_B) # for test

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss = ops_UNIT.KL_divergence(shared)
        # enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        # enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        # l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab

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
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()


class UNIT_MultiEncSpecificBranchFromImg_CycleGAN_ChangeRes_FeaMask_VggStyleContentLoss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_VggStyleContentLoss):
    def translation(self, x_A, x_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B):
        ## Common branch
        out = tf.concat([self.encoder(x_A, self.is_training, scope="encoder_A"), self.encoder(x_B, self.is_training, scope="encoder_B")], axis=0)
        # shared = self.share_encoder(out, self.is_training)
        # out = self.share_generator(shared, self.is_training)
        ## Specific branch

        out_A = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_A, beta_list_A, self.is_training, scope="generator_A")
        out_B = self.generator_two(out, feaMap_list_A, feaMap_list_B, gamma_list_B, beta_list_B, self.is_training, scope="generator_B")

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        return x_Aa, x_Ba, x_Ab, x_Bb #, shared

    def generate_a2b(self, x_A, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_A, self.is_training, reuse=True, scope="encoder_A")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_B")

        return out #, shared

    def generate_b2a(self, x_B, feaMap_list=None, gamma_list=None, beta_list=None):
        out = self.encoder(x_B, self.is_training, reuse=True, scope="encoder_B")
        # shared = self.share_encoder(out, self.is_training, reuse=True)
        # out = self.share_generator(shared, self.is_training, reuse=True)
        out = self.generator_one(out, feaMap_list, gamma_list, beta_list, self.is_training, reuse=True, scope="generator_A")

        return out #, shared

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
            x_aa, x_ba, x_ab, x_bb = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
            x_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
            x_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
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
            self.fake_B = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
            self.fake_A = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
            self.fake_B_feaMasked = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
            self.fake_A_feaMasked = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
            self.fake_B_insNormed = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
            self.fake_A_insNormed = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

        """ Define Loss """
        G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
        G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

        D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
        D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

        # enc_loss = ops_UNIT.KL_divergence(shared)
        # enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
        # enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

        # l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
        # l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
        l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
        l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

        content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)
        self.content_loss, self.style_loss = content_loss, style_loss

        Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
                           self.L1_cycle_weight * l1_loss_aba

        Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
                           self.L1_cycle_weight * l1_loss_bab

        Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
        Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 5e2*content_loss
        # self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e4*style_loss + 1e2*content_loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss + 1e3*style_loss + 1e2*content_loss
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
        self.lr_sum = tf.summary.scalar("lr", self.lr)
        self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
        self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
        self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
        self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
        self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)

        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge([self.lr_sum, self.G_A_loss, self.G_B_loss, self.all_G_loss,
                        self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, 
                        self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.content_loss_sum, self.style_loss_sum])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        self.init_op = tf.global_variables_initializer()
        self._define_output()


# ## Combined with pretrained seg model
# from FCN import fcn8_vgg
# class UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss_FCN8Loss(UNIT_MultiEncSpecificBranchFromImg_Cycle_ChangeRes_FeaMask_EncStyleContentLoss):
#     def init_net(self, args):
#         assert args.pretrained_path is not None
#         if args.pretrained_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_A') \
#                 + tf.get_collection(tf.GraphKeys.VARIABLES, scope='VAEGAN_recon_B')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverPart = tf.train.Saver(var, max_to_keep=5)
#         if args.pretrained_fcn_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg16_fcn8')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverFcnPart = tf.train.Saver(var, max_to_keep=5)
#         if args.pretrained_vgg_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='vgg_19')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverVggPart = tf.train.Saver(var, max_to_keep=5)
#         if args.test_model_path is not None:
#             var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='UNIT')
#             var = var_filter_by_exclude(var, exclude_scopes=['Adam'])
#             self.saverTest = tf.train.Saver(var, max_to_keep=5)
#         self.summary_writer = tf.summary.FileWriter(args.model_dir)
#         sv = tf.train.Supervisor(logdir=args.model_dir, is_chief=True, saver=None, summary_op=None, 
#                 summary_writer=self.summary_writer, save_model_secs=0, ready_for_local_init_op=None)
#         if args.phase == 'train':
#             gpu_options = tf.GPUOptions(allow_growth=True)
#         else:
#             gpu_options = tf.GPUOptions(allow_growth=True)
#         sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#         self.sess = sv.prepare_or_wait_for_session(config=sess_config)

#     # ##############################################################################
#     # # BEGIN of ENCODERS
#     # def encoder(self, x, is_training=True, reuse=False, scope="encoder"):
#     #     channel = self.ngf
#     #     with tf.variable_scope(scope, reuse=reuse) :
#     #         x = ops_UNIT.conv(x, channel, kernel=7, stride=1, pad=3, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_0')

#     #         for i in range(1, self.n_encoder) :
#     #             x = ops_UNIT.conv(x, channel*2, kernel=3, stride=2, pad=1, normal_weight_init=self.normal_weight_init, activation_fn='leaky', scope='conv_'+str(i))
#     #             channel *= 2

#     #         # channel = 256
#     #         for i in range(0, self.n_enc_resblock) :
#     #             x = ops_UNIT.resblock(x, channel, kernel=3, stride=1, pad=1, dropout_ratio=self.res_dropout,
#     #                          normal_weight_init=self.normal_weight_init,
#     #                          is_training=is_training, norm_fn=self.norm, scope='resblock_'+str(i))

#     #         return x
#     # # END of ENCODERS
#     # ##############################################################################

#     def _build_model(self):
#         self._define_input()
#         domain_A = self.domain_A
#         domain_B = self.domain_B

#         #### image translation ####
#         feaMap_list_A, gamma_list_A, beta_list_A = self.ins_specific_branch(domain_A, is_training=False, scope='VAEGAN_recon_A')
#         feaMap_list_B, gamma_list_B, beta_list_B = self.ins_specific_branch(domain_B, is_training=False, scope='VAEGAN_recon_B')
#         self.feaMask_list_A = [tf.sigmoid(feaMap_list_A[i]) for i in range(len(feaMap_list_A))]
#         self.feaMask_list_B = [tf.sigmoid(feaMap_list_B[i]) for i in range(len(feaMap_list_B))]
#         with tf.variable_scope('UNIT'):
#             """ Define Encoder, Generator, Discriminator """
#             """feaMap_list should be consistent with original x, eg. x_ba should be maskded with feaMap_list_B """
#             x_aa, x_ba, x_ab, x_bb, shared = self.translation(domain_A, domain_B, feaMap_list_A, gamma_list_A, beta_list_A, feaMap_list_B, gamma_list_B, beta_list_B)
#             x_bab, shared_bab = self.generate_a2b(x_ba, feaMap_list_B, gamma_list_B, beta_list_B)
#             x_aba, shared_aba = self.generate_b2a(x_ab, feaMap_list_A, gamma_list_A, beta_list_A)
#             self.x_aa, self.x_ba, self.x_ab, self.x_bb = x_aa, x_ba, x_ab, x_bb

#             # real_A_logit, real_B_logit = self.discriminate_real(domain_A, domain_B)  
#             real_A_logit = self.discriminator(domain_A, scope="discriminator_A")
#             real_B_logit = self.discriminator(domain_B, scope="discriminator_B")

#             if self.replay_memory :
#                 self.fake_A_pool = ImagePool_UNIT(self.pool_size)  # pool of generated A
#                 self.fake_B_pool = ImagePool_UNIT(self.pool_size)  # pool of generated B
#                 # fake_A_logit, fake_B_logit = self.discriminate_fake_pool(x_ba, x_ab)
#                 fake_A_logit = self.discriminator(self.fake_A_pool.query(x_ba), reuse=True, scope="discriminator_A") # replay memory
#                 fake_B_logit = self.discriminator(self.fake_B_pool.query(x_ab), reuse=True, scope="discriminator_B") # replay memory
#             else :
#                 # fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
#                 fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
#                 fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

#             """ Generated Image """
#             self.fake_B, _ = self.generate_a2b(domain_A, None, None, None) # for test without applying Instance Norm
#             self.fake_A, _ = self.generate_b2a(domain_B, None, None, None) # for test without applying Instance Norm
#             self.fake_B_feaMasked, _ = self.generate_a2b(domain_A, feaMap_list_A, None, None) # for test without applying Instance Norm
#             self.fake_A_feaMasked, _ = self.generate_b2a(domain_B, feaMap_list_B, None, None) # for test without applying Instance Norm
#             self.fake_B_insNormed, _ = self.generate_a2b(domain_A, None, gamma_list_B, beta_list_B) # for test without applying Instance Norm
#             self.fake_A_insNormed, _ = self.generate_b2a(domain_B, None, gamma_list_A, beta_list_A) # for test without applying Instance Norm

#         #### Extract segmentation ####
#         self.seg_model = fcn8_vgg.FCN8VGG(vgg16_npy_path='./weights/vgg16.npy')
#         input_data = tf.concat([domain_A, domain_B, x_ab, x_ba], axis=0)
#         with tf.variable_scope('vgg16_fcn8') as vs:
#             self.seg_model.build(input_data, train=True, num_classes=self.segment_class, random_init_fc8=True, debug=False, use_dilated=False)
#         self.pred_mask_prob_A, self.pred_mask_prob_B, self.pred_mask_prob_ab, self.pred_mask_prob_ba = tf.split(self.seg_model.upscore32, 4, axis=0)
#         self.pred_mask_A = tf.argmax(self.pred_mask_prob_A[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_B = tf.argmax(self.pred_mask_prob_B[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_ab = tf.argmax(self.pred_mask_prob_ab[:,:,:,0:self.segment_class-1], -1)
#         self.pred_mask_ba = tf.argmax(self.pred_mask_prob_ba[:,:,:,0:self.segment_class-1], -1)
#         if 8==self.segment_class:
#             self.pred_mask_A += 1
#             self.pred_mask_B += 1
#             self.pred_mask_ab += 1
#             self.pred_mask_ba += 1

#         """ Define Loss """
#         G_ad_loss_a = ops_UNIT.generator_loss(fake_A_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)
#         G_ad_loss_b = ops_UNIT.generator_loss(fake_B_logit, smoothing=self.smoothing, use_lsgan=self.use_lsgan)

#         D_ad_loss_a = ops_UNIT.discriminator_loss(real_A_logit, fake_A_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)
#         D_ad_loss_b = ops_UNIT.discriminator_loss(real_B_logit, fake_B_logit, smoothing=self.smoothing, use_lasgan=self.use_lsgan)

#         enc_loss = ops_UNIT.KL_divergence(shared)
#         enc_bab_loss = ops_UNIT.KL_divergence(shared_bab)
#         enc_aba_loss = ops_UNIT.KL_divergence(shared_aba)

#         l1_loss_a = ops_UNIT.L1_loss(x_aa, domain_A) # identity
#         l1_loss_b = ops_UNIT.L1_loss(x_bb, domain_B) # identity
#         l1_loss_aba = ops_UNIT.L1_loss(x_aba, domain_A) # reconstruction
#         l1_loss_bab = ops_UNIT.L1_loss(x_bab, domain_B) # reconstruction

#         content_loss, style_loss = self.encoder_style_content_loss(domain_A, domain_B, x_ab, x_ba, is_training=False, reuse=tf.AUTO_REUSE)

#         seg_consis_loss_a = tf.reduce_mean(ops_UNIT.L2_loss(self.pred_mask_prob_A, self.pred_mask_prob_ab)) # semantic consistency
#         seg_consis_loss_b = tf.reduce_mean(ops_UNIT.L2_loss(self.pred_mask_prob_B, self.pred_mask_prob_ba)) # semantic consistency
#         # seg_consis_loss = seg_consis_loss_a + seg_consis_loss_b
#         seg_consis_loss = seg_consis_loss_a

#         Generator_A_loss = self.GAN_weight * G_ad_loss_a + \
#                            self.L1_weight * l1_loss_a + \
#                            self.L1_cycle_weight * l1_loss_aba + \
#                            self.KL_weight * enc_loss + \
#                            self.KL_cycle_weight * enc_bab_loss

#         Generator_B_loss = self.GAN_weight * G_ad_loss_b + \
#                            self.L1_weight * l1_loss_b + \
#                            self.L1_cycle_weight * l1_loss_bab + \
#                            self.KL_weight * enc_loss + \
#                            self.KL_cycle_weight * enc_aba_loss

#         Discriminator_A_loss = self.GAN_weight * D_ad_loss_a
#         Discriminator_B_loss = self.GAN_weight * D_ad_loss_b

#         self.Generator_loss = Generator_A_loss + Generator_B_loss + 0.*style_loss + 0.*content_loss + 1.*seg_consis_loss
#         self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
#         self.D_loss_A = Discriminator_A_loss
#         self.D_loss_B = Discriminator_B_loss


#         """ Training """
#         t_vars = tf.trainable_variables()
#         G_vars = [var for var in t_vars if ('generator' in var.name) or ('encoder' in var.name)]
#         D_vars = [var for var in t_vars if 'discriminator' in var.name]


#         # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#         # pdb.set_trace()
#         self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
#         self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

#         """" Summary """
#         self.G_ad_loss_a_sum = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
#         self.G_ad_loss_b_sum = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)
#         self.l1_loss_a_sum = tf.summary.scalar("l1_loss_a", l1_loss_a)
#         self.l1_loss_b_sum = tf.summary.scalar("l1_loss_b", l1_loss_b)
#         self.l1_loss_aba_sum = tf.summary.scalar("l1_loss_aba", l1_loss_aba)
#         self.l1_loss_bab_sum = tf.summary.scalar("l1_loss_bab", l1_loss_bab)
#         self.enc_loss_sum = tf.summary.scalar("KL_enc_loss", enc_loss)
#         self.enc_bab_loss_sum = tf.summary.scalar("KL_enc_bab_loss", enc_bab_loss)
#         self.enc_aba_loss_sum = tf.summary.scalar("KL_enc_aba_loss", enc_aba_loss)
#         self.content_loss_sum = tf.summary.scalar("content_loss", content_loss)
#         self.style_loss_sum = tf.summary.scalar("style_loss", style_loss)
#         self.seg_consis_loss_a_sum = tf.summary.scalar("seg_consis_loss_a", seg_consis_loss_a)
#         self.seg_consis_loss_b_sum = tf.summary.scalar("seg_consis_loss_b", seg_consis_loss_b)

#         self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
#         self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
#         self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
#         self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
#         self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
#         self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

#         self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss,
#                         self.G_ad_loss_a_sum, self.G_ad_loss_b_sum, self.l1_loss_a_sum, self.l1_loss_b_sum,
#                         self.l1_loss_aba_sum, self.l1_loss_bab_sum, self.enc_loss_sum,
#                         self.enc_bab_loss_sum, self.enc_aba_loss_sum, 
#                         self.content_loss_sum, self.style_loss_sum, self.seg_consis_loss_a_sum, self.seg_consis_loss_b_sum])
#         self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

#         self.init_op = tf.global_variables_initializer()
#         self._define_output()

#     def sample_model(self, sample_dir, epoch, idx):
#         img_name_A, img_name_B, real_A, real_B, fake_A, fake_B, fake_A_feaMasked, fake_B_feaMasked, fake_A_insNormed, fake_B_insNormed, x_aa, x_ba, x_ab, x_bb, feaMapA, feaMapB, \
#             pred_mask_A, pred_mask_B, pred_mask_ab, pred_mask_ba = self.sess.run(
#                 [self.img_name_A, self.img_name_B, self.real_A, self.real_B, self.fake_A, self.fake_B, \
#                 self.fake_A_feaMasked, self.fake_B_feaMasked, self.fake_A_insNormed, self.fake_B_insNormed, \
#                 self.x_aa, self.x_ba, self.x_ab, self.x_bb, self.feaMask_list_A[0], self.feaMask_list_B[0], \
#                 self.pred_mask_A, self.pred_mask_B, self.pred_mask_ab, self.pred_mask_ba], \
#                 feed_dict={self.is_training : False}
#             )

#         real_A_img = unprocess_image(real_A, 127.5, 127.5)
#         real_B_img = unprocess_image(real_B, 127.5, 127.5)
#         fake_A_img = unprocess_image(fake_A, 127.5, 127.5)
#         fake_B_img = unprocess_image(fake_B, 127.5, 127.5)
#         fake_A_feaMasked_img = unprocess_image(fake_A_feaMasked, 127.5, 127.5)
#         fake_B_feaMasked_img = unprocess_image(fake_B_feaMasked, 127.5, 127.5)
#         fake_A_insNormed_img = unprocess_image(fake_A_insNormed, 127.5, 127.5)
#         fake_B_insNormed_img = unprocess_image(fake_B_insNormed, 127.5, 127.5)
#         x_aa_img = unprocess_image(x_aa, 127.5, 127.5)
#         x_ba_img = unprocess_image(x_ba, 127.5, 127.5)
#         x_ab_img = unprocess_image(x_ab, 127.5, 127.5)
#         x_bb_img = unprocess_image(x_bb, 127.5, 127.5)
#         img_name_A = img_name_A[0]
#         img_name_B = img_name_B[0]
#         save_images(real_A_img, [self.batch_size, 1],
#                     '{}/real_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(real_B_img, [self.batch_size, 1],
#                     '{}/real_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_img, [self.batch_size, 1],
#                     '{}/fake_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_img, [self.batch_size, 1],
#                     '{}/fake_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_feaMasked_img, [self.batch_size, 1],
#                     '{}/fake_A_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_feaMasked_img, [self.batch_size, 1],
#                     '{}/fake_B_feaMasked_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(fake_A_insNormed_img, [self.batch_size, 1],
#                     '{}/fake_A_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(fake_B_insNormed_img, [self.batch_size, 1],
#                     '{}/fake_B_insNormed_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(x_aa_img, [self.batch_size, 1],
#                     '{}/x_aa_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(x_ba_img, [self.batch_size, 1],
#                     '{}/x_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         save_images(x_ab_img, [self.batch_size, 1],
#                     '{}/x_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_B.split(".")[0]))
#         save_images(x_bb_img, [self.batch_size, 1],
#                     '{}/x_bb_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

#         for i in range(self.batch_size):
#             # pdb.set_trace()
#             fake_A_tmp, fake_B_tmp, x_ba_tmp, x_ab_tmp = self.sess.run(
#                 [self.fake_A, self.fake_B, self.x_ba, self.x_ab], \
#                 feed_dict={self.is_training : False, 
#                             self.real_A : np.tile(real_A[i,:,:,:], [self.batch_size, 1, 1, 1]), 
#                             self.real_B : real_B}
#             )
#             fake_A_tmp_img = unprocess_image(fake_A_tmp, 127.5, 127.5)
#             fake_B_tmp_img = unprocess_image(fake_B_tmp, 127.5, 127.5)
#             x_ba_tmp_img = unprocess_image(x_ba_tmp, 127.5, 127.5)
#             x_ab_tmp_img = unprocess_image(x_ab_tmp, 127.5, 127.5)
#             save_images(fake_A_tmp_img, [self.batch_size, 1],
#                         '{}/fake_A_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(fake_B_tmp_img, [self.batch_size, 1],
#                         '{}/fake_B_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(x_ba_tmp_img, [self.batch_size, 1],
#                         '{}/x_ba_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
#             save_images(x_ab_tmp_img, [self.batch_size, 1],
#                         '{}/x_ab_InsNorm{}_{:02d}_{:04d}_{}.png'.format(sample_dir, i, epoch, idx, img_name_B.split(".")[0]))
            
#         def vis_proc_func(mask):
#             # catId2color = { label.categoryId: label.color for label in labels }
#             trainId2color = { label.trainId: label.color for label in labels }
#             trainId2color[19] = trainId2color[255]
#             ## TODO: seg logits --> one_hot --> catId --> color
#             # seg_catId = seg.argmax(axis=-1)
#             batch_size, img_h, img_w = mask.shape
#             seg_rgb = np.zeros([batch_size, img_h, img_w, 3])
#             for b in range(batch_size):
#                 for h in range(img_h):
#                     for w in range(img_w):
#                         # seg_rgb[b,h,w,:] = catId2color[mask[b,h,w]]
#                         seg_rgb[b,h,w,:] = trainId2color[mask[b,h,w]]
#             return seg_rgb
#         imsave(vis_proc_func(pred_mask_A), [self.batch_size, 1],
#                     '{}/Seg_A_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_B), [self.batch_size, 1],
#                     '{}/Seg_B_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_ab), [self.batch_size, 1],
#                     '{}/Seg_ab_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))
#         imsave(vis_proc_func(pred_mask_ba), [self.batch_size, 1],
#                     '{}/Seg_ba_{:02d}_{:04d}_{}.png'.format(sample_dir, epoch, idx, img_name_A.split(".")[0]))

#     def train(self, args):
#         """Train SG-GAN"""
#         self.sess.run(self.init_op)
#         self.writer = tf.summary.FileWriter(args.model_dir, self.sess.graph)
#         if args.pretrained_path is not None:
#             self.saverPart.restore(self.sess, args.pretrained_path)
#             print('restored from pretrained_path:', args.pretrained_path)
#         if args.pretrained_fcn_path is not None:
#             self.saverFcnPart.restore(self.sess, args.pretrained_fcn_path)
#             print('restored from pretrained_fcn_path:', args.pretrained_fcn_path)
#         if args.pretrained_vgg_path is not None:
#             self.saverVggPart.restore(self.sess, args.pretrained_vgg_path)
#             print('restored from pretrained_vgg_path:', args.pretrained_vgg_path)

#         counter = args.global_step
#         start_time = time.time()

#         if 0==args.global_step:
#             if args.continue_train and self.load_last_ckpt(args.model_dir):
#                 print(" [*] Load SUCCESS")
#             else:
#                 print(" [!] Load failed...")
#         else:
#             ## global_step is set manually
#             if args.continue_train and self.load_ckpt(args.model_dir, args.global_step):
#                 print(" [*] Load SUCCESS")
#             else:
#                 print(" [!] Load failed...")

#         for epoch in range(args.epoch):
#             # batch_idxs = self.sample_num // self.batch_size
#             batch_idxs = self.sample_num
#             # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
#             # idx = int(epoch/len(self.weights_schedule))
#             # loss_weights = self.weights_schedule[idx]

#             d_loss = np.inf
#             for idx in range(0, batch_idxs):
#                 step_ph = epoch*batch_idxs + idx
#                 num_steps = args.epoch*batch_idxs
#                 lr = args.lr*((1 - counter / num_steps)**0.9)
#                 # Update D
#                 # if d_loss>3.0:
#                 if self.D_optim is not None:
#                     _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], \
#                                                             feed_dict={self.is_training : True, self.lr : lr})
#                     self.writer.add_summary(summary_str, counter)

#                 # Update G
#                 fake_A, fake_B, _, g_loss, summary_str = self.sess.run([self.fake_A, self.fake_B, self.G_optim, self.Generator_loss, self.G_loss], \
#                                                                         feed_dict={self.is_training : True, self.lr : lr})
#                 self.writer.add_summary(summary_str, counter)

#                 counter += 1
#                 if np.mod(counter, args.print_freq) == 1:
#                     # display training status
#                     print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
#                           % (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss))
#                     self.sample_model(args.sample_dir, epoch, counter)

#                 if (counter>3000 and np.mod(counter, args.save_freq) == 2) or (idx==batch_idxs-1):
#                     self.save(args.model_dir, counter)