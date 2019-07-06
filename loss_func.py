from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *


def LOSS_L1(target, pred):
    return tf.reduce_mean(tf.abs(target - pred))

def LOSS_L2(target, pred):
    return tf.sqrt(tf.reduce_mean(tf.square(target - pred)))

def LOSS_huber(target, pred):
    return tf.losses.huber_loss(tf.log(0.01+target), tf.log(0.01+pred), delta=0.2)

## log diff loss. ref to https://github.com/MasazI/cnn_depth_tensorflow/blob/master/model.py
def LOSS_log_scale_invariant(target, pred_disp):
    height, width = target.get_shape()[1].value, target.get_shape()[2].value
    # pdb.set_trace()
    pix_num = float(height*width)
    d = tf.log(0.01+pred_disp) - tf.log(0.01+target)
    # d = tf.log(1+pred_disp) - tf.log(1+target)
    # d = pred_disp - target
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / pix_num - 0.5*sqare_sum_d / math.pow(pix_num, 2))
    return cost

## Smooth loss. ref to https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy
    
def depth_smoothness(img, disps):
    pyramid_imgs  = scale_pyramid(img,  len(disps))
    depth_gradients_x = [gradient_x(d) for d in disps]
    depth_gradients_y = [gradient_y(d) for d in disps]

    image_gradients_x = [gradient_x(img) for img in pyramid_imgs]
    image_gradients_y = [gradient_y(img) for img in pyramid_imgs]

    weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
    weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

    smoothness_x = [depth_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [depth_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y

def LOSS_depth_gradient(img, pred_disps):
    pred_depth_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in pred_disps]
    pred_depth_smoothness  = depth_smoothness(img, pred_depth_est)
    depth_gradient_losses = [tf.reduce_mean(tf.abs(pred_depth_smoothness[i])) / 2 ** i for i in range(len(pred_disps))]
    depth_gradient_loss = tf.add_n(depth_gradient_losses)
    return depth_gradient_loss

## SSIM loss. ref to https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
def LOSS_SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

## ref to https://github.com/kwotsin/TensorFlow-ENet/blob/master/train_enet.py
def LOSS_semantic_seg_task(pred_mask_list, target, loss_weights, weight_decay, scale_num=4, segment_class=8):
    # pdb.set_trace()
    ## Perform one-hot-encoding on the ground truth annotation to get same shape as the logits
    target = tf.one_hot(target, segment_class, axis=-1)

    scale_num = len(pred_mask_list)
    height, width = target.get_shape()[1].value, target.get_shape()[2].value
    t_vars = tf.trainable_variables()
    depth_net_vars = [var for var in t_vars if 'seg_depth_net' in var.name]
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with tf.name_scope("seg_losses"):
        target_list = [tf.image.resize_nearest_neighbor(target, [int(height / np.power(2, n)), int(width / np.power(2, n))])
                        for n in range(scale_num)]

        ####### Weighted Spatial softmax cross extropy ############ 
        # The class_weights list can be multiplied by onehot_labels directly because the last dimension
        # of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want. 
        # Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
        # to get a mask with each pixel's value representing the class_weight.
        ###########################################################
        # weights = onehot_labels * class_weights
        # weights = tf.reduce_sum(weights, 3)
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)
        Seg_losses = [tf.losses.softmax_cross_entropy(onehot_labels=target_list[n], logits=pred_mask_list[n])
                        for n in range(scale_num)]
        for i in range(scale_num):
            tf.summary.scalar('Seg_loss' + str(i), Seg_losses[i])
            tf.summary.scalar('Seg_loss_weight' + str(i), loss_weights[i])
        Seg_loss = tf.add_n([Seg_losses[i]*loss_weights[i] for i in range(scale_num)])

        reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=depth_net_vars)
        total_loss = Seg_loss*1.0 + reg_loss*0.0
        tf.summary.scalar('Seg_loss', Seg_loss)
        # tf.summary.scalar('total_loss', total_loss)
        error = Seg_losses[0]
        return total_loss, Seg_loss, error

## ref to https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
def LOSS_depth_task(img, pred_depth_list, target, loss_weights, weight_decay, scale_num=4):
    height, width = target.get_shape()[1].value, target.get_shape()[2].value
    t_vars = tf.trainable_variables()
    depth_net_vars = [var for var in t_vars if 'depth_net' in var.name]
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with tf.name_scope("depth_losses"):
        target_list = [tf.image.resize_nearest_neighbor(target, [int(height / np.power(2, n)), int(width / np.power(2, n))])
                        for n in range(scale_num)]
        ## L1 loss
        LSI_losses = [LOSS_log_scale_invariant(target_list[i], pred_depth_list[i]) for i in range(scale_num)]
        # LSI_losses = [LOSS_huber(target_list[i], pred_depth_list[i]) for i in range(scale_num)]
        for i in range(scale_num):
            tf.summary.scalar('L1_loss' + str(i), LSI_losses[i])
            tf.summary.scalar('L1_loss_weight' + str(i), loss_weights[i])
        LSI_loss = tf.add_n([LSI_losses[i]*loss_weights[i] for i in range(scale_num)])


        ## Depth gradient loss
        depth_gradient_loss = LOSS_depth_gradient(img, pred_depth_list)

        reg_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=depth_net_vars)
        total_loss = LSI_loss*1.0 + depth_gradient_loss*0.001 + reg_loss*0.0
        tf.summary.scalar('LSI_loss', LSI_loss)
        tf.summary.scalar('depth_gradient_loss', depth_gradient_loss)
        # tf.summary.scalar('total_loss', total_loss)
        error = LSI_losses[0]
        return total_loss, LSI_loss, depth_gradient_loss, error