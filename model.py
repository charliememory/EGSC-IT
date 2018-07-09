from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)

def tf_deriv(batch, ksize=3, padding='SAME'):
    n_ch = int(batch.get_shape().as_list()[3])
    gx = tf_kernel_prep_3d(np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]), n_ch)
    gy = tf_kernel_prep_3d(np.array([[-1,-2, -1],
                                     [ 0, 0, 0],
                                     [ 1, 2, 1]]), n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel_image", dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")

def discriminator(image, mask, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 512 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 256 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 128 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 64 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 64 x self.df_dim*8)
        h4 = conv2d(h3, options.segment_class, s=1, name='d_h4_conv')
        h4_mask = tf.reduce_sum(tf.multiply(h4, mask), axis=-1, keep_dims=True)
        # h4_mask is (32 x 64 x 1)
        return h4_mask

def discriminator_original(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 512 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 256 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 128 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 64 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 64 x self.df_dim*8)
        # h4 = conv2d(h3, options.segment_class, s=1, name='d_h4_conv')
        h4 = conv2d(h3, 1, s=1, name='d_h4_conv')
        # h4_mask = tf.reduce_sum(tf.multiply(h4, mask), axis=-1, keep_dims=True)
        # h4_mask is (32 x 64 x 1)
        return h4

def discriminator_sharePart(image, mask, options, reuse1=True, reuse2=False, name="discriminator"):

    with tf.variable_scope(name):
        with tf.variable_scope('share', reuse=reuse1):

            h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
            # h0 is (128 x 256 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 128 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 64 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
            # h3 is (32 x 64 x self.df_dim*8)
        if mask is not None:
            with tf.variable_scope('masked_dis_out', reuse=reuse2):
                h4 = conv2d(h3, options.segment_class, s=1, name='d_h4_conv')
                h4_mask = tf.reduce_sum(tf.multiply(h4, mask), axis=-1, keep_dims=True)
                # h4_mask is (32 x 64 x 8)
        else:
            with tf.variable_scope('dis_out', reuse=reuse2):
                h4 = conv2d(h3, 1, s=1, name='d_h4_conv')
                # h4_mask is (32 x 64 x 1)
        return h4

def generator_unet_norm(image, options, reuse=False, name="generator"):

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')

        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')

        return tf.nn.tanh(d8)

def generator_unet(image, options, reuse=False, name="generator"):
    b, h, w, _ = image.get_shape().as_list()
    final_stride=2 if h%256==0 else 1

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        e1 = conv2d(image, options.gf_dim, name='g_e1_conv')
        e2 = conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv')
        e3 = conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv')
        e4 = conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv')
        e5 = conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv')
        e6 = conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv')

        e7 = conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv')
        e8 = conv2d(lrelu(e7), options.gf_dim*8, s=final_stride, name='g_e8_conv')

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, s=final_stride, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([d1, e7], 3)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([d2, e6], 3)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([d3, e5], 3)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([d4, e4], 3)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([d5, e3], 3)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([d6, e2], 3)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([d7, e1], 3)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')

        return tf.nn.tanh(d8)

def generator_resnet_norm(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        # For 256*832 img, add one more conv2d and deconve2d
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*3, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim*4, 3, 2, name='g_e4_c'), 'g_e4_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c4, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*3, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim*2, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d3 = deconv2d(d2, options.gf_dim, 3, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))
        d3 = tf.pad(d3, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d3, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'))
        c2 = tf.nn.relu(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'))
        c3 = tf.nn.relu(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(d1)
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(d2)
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def generator_resEnc(image, options, reuse=False, use_deconv=False, name="generator"):
    with tf.variable_scope(name, reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(image, options.gf_dim, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1, options.gf_dim, 1) # H/8  -  256D
            conv3 = resblock(conv2, options.gf_dim*2, 1) # H/16 -  512D
            conv4 = resblock(conv3, options.gf_dim*3, 1) # H/32 - 1024D
            conv5 = resblock(conv4, options.gf_dim*4, 1) # H/64 - 2048D
            # conv6 = resblock(conv5, options.gf_dim*5, 1) # H/128 - 2048D
            # conv7 = resblock(conv6, options.gf_dim*6, 1) # H/256 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            # skip6 = conv5
            # skip7 = conv6
        
        # DECODING
        with tf.variable_scope('decoder'):
            # upconv8 = upconv_func(conv7, options.gf_dim*6, 3, 2) #H/128
            # concat8 = tf.concat([upconv8, skip7], 3)
            # iconv8  = conv_nn(concat8, options.gf_dim*6, 3, 1)

            # upconv7 = upconv_func(iconv8, options.gf_dim*5, 3, 2) #H/64
            # concat7 = tf.concat([upconv7, skip6], 3)
            # iconv7  = conv_nn(concat7, options.gf_dim*5, 3, 1)

            # upconv6 = upconv_func(iconv7, options.gf_dim*4, 3, 2) #H/32
            upconv6 = upconv_func(conv5, options.gf_dim*4, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv_nn(concat6, options.gf_dim*4, 3, 1)

            upconv5 = upconv_func(iconv6, options.gf_dim*3, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv_nn(concat5, options.gf_dim*3, 3, 1)

            upconv4 = upconv_func(iconv5, options.gf_dim*2, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv_nn(concat4, options.gf_dim*2, 3, 1)

            upconv3 = upconv_func(iconv4, options.gf_dim, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2], 3)
            iconv3  = conv_nn(concat3, options.gf_dim, 3, 1)

            upconv2 = upconv_func(iconv3, options.gf_dim/2, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1], 3)
            iconv2  = conv_nn(concat2, options.gf_dim/2, 3, 1)

            upconv1 = upconv_func(iconv2, options.gf_dim/4, 3, 2) #H
            iconv1  = conv_nn(upconv1, options.gf_dim/4, 3, 1)

            # pdb.set_trace()
            pred  = conv_nn(iconv1,   3, 3, 1)

    return pred

def generator_resEncDec(image, options, reuse=False, use_deconv=False, name="generator"):
    with tf.variable_scope(name, reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(image, options.gf_dim, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1, options.gf_dim, 1) # H/8  -  256D
            conv3 = resblock(conv2, options.gf_dim*2, 1) # H/16 -  512D
            conv4 = resblock(conv3, options.gf_dim*3, 1) # H/32 - 1024D
            conv5 = resblock(conv4, options.gf_dim*4, 1) # H/64 - 2048D
            # conv6 = resblock(conv5, options.gf_dim*5, 1) # H/128 - 2048D
            # conv7 = resblock(conv6, options.gf_dim*6, 1) # H/256 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
            # skip6 = conv5
            # skip7 = conv6
        
        # DECODING
        with tf.variable_scope('decoder'):
            # upconv8 = upconv_func(conv7, options.gf_dim*6, 3, 2) #H/128
            # concat8 = tf.concat([upconv8, skip7], 3)
            # iconv8  = conv_nn(concat8, options.gf_dim*6, 3, 1)

            # upconv7 = upconv_func(iconv8, options.gf_dim*5, 3, 2) #H/64
            # concat7 = tf.concat([upconv7, skip6], 3)
            # iconv7  = conv_nn(concat7, options.gf_dim*5, 3, 1)

            # upconv6 = upconv_func(iconv7, options.gf_dim*4, 3, 2) #H/32
            upconv6 = upconv_func(conv5, options.gf_dim*4, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = resblock(concat6, options.gf_dim*4, 1, down_sample=False)

            upconv5 = upconv_func(iconv6, options.gf_dim*3, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = resblock(concat5, options.gf_dim*3, 1, down_sample=False)

            upconv4 = upconv_func(iconv5, options.gf_dim*2, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = resblock(concat4, options.gf_dim*2, 1, down_sample=False)

            upconv3 = upconv_func(iconv4, options.gf_dim, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2], 3)
            iconv3  = resblock(concat3, options.gf_dim, 1, down_sample=False)

            upconv2 = upconv_func(iconv3, options.gf_dim/2, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1], 3)
            iconv2  = resblock(concat2, options.gf_dim/2, 1, down_sample=False)

            upconv1 = upconv_func(iconv2, options.gf_dim/4, 3, 2) #H
            iconv1  = resblock(upconv1, options.gf_dim/4, 1, down_sample=False)

            # pdb.set_trace()
            pred  = conv_nn(iconv1,   3, 3, 1)

    return pred


def generator_UAEAfterResidual_PG2(image, options, reuse=False, use_deconv=False, repeat_num=5, activation_fn=tf.nn.elu, name="generator"):
    with tf.variable_scope(name, reuse=reuse) as sc:
        input_channel  = image.get_shape().as_list()[-1]
        hidden_num = options.gf_dim
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(image, hidden_num, 3, 1, activation_fn=activation_fn)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = x + res
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn)
        
        for idx in range(repeat_num):
            x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            res = x
            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = x.get_shape()[-1]
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = x + res
            if idx < repeat_num - 1:
                # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn)
                x = upscale(x, 2)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn)

        # pdb.set_trace()
        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None)

    # variables = tf.contrib.framework.get_variables(vs)
    return out


def generator_UAEAfterResidual_PG2_diffMap(image, options, reuse=False, use_deconv=False, repeat_num=5, activation_fn=tf.nn.elu, name="generator"):
    diffMap = generator_UAEAfterResidual_PG2(image, options, reuse, use_deconv, repeat_num, activation_fn, name)
    out = image + diffMap
    # variables = tf.contrib.framework.get_variables(vs)
    return out

def generator_UAEAfterResidual_PG2_wxb(image, options, reuse=False, use_deconv=False, repeat_num=5, activation_fn=tf.nn.elu, name="generator"):
    with tf.variable_scope(name, reuse=reuse) as sc:
        input_channel  = image.get_shape().as_list()[-1]
        hidden_num = options.gf_dim
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(image, hidden_num, 3, 1, activation_fn=activation_fn)

        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
            x = x + res
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn)
        ## Weight
        w = x
        for idx in range(repeat_num-1):
            # w = slim.layers.conv2d_transpose(w, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn)
            w = upscale(w, 2)
            w = slim.conv2d(w, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn)
        w = slim.conv2d(w, input_channel, 3, 1, activation_fn=None)
        ## Bias
        b = x
        for idx in range(repeat_num):
            b = tf.concat([b, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            res = b
            # channel_num = hidden_num * (repeat_num-idx)
            channel_num = b.get_shape()[-1]
            b = slim.conv2d(b, channel_num, 3, 1, activation_fn=activation_fn)
            b = slim.conv2d(b, channel_num, 3, 1, activation_fn=activation_fn)
            b = b + res
            if idx < repeat_num - 1:
                # b = slim.layers.conv2d_transpose(b, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn)
                b = upscale(b, 2)
                b = slim.conv2d(b, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn)
        b = slim.conv2d(b, input_channel, 3, 1, activation_fn=None)

        out = tf.multiply(1.0+w, image) + b

    # variables = tf.contrib.framework.get_variables(vs)
    return out


def generator_UAEAfterResidual_PG2_imgseg(imgSrcDom, maskSrcDom, imgDstDom, options, reuse1=False, reuse2=False, use_deconv=False, repeat_num=5, activation_fn=tf.nn.elu, name="generator"):
    with tf.variable_scope(name) as sc:
        hidden_num = options.gf_dim
        with tf.variable_scope('weight_sharing_1', reuse=reuse1) as sc:
            if (imgSrcDom is not None) and (maskSrcDom is not None) and (imgDstDom is None):
                ## Utilize the seg label
                input_channel  = imgSrcDom.get_shape().as_list()[-1]
                b, h, w, c = imgSrcDom.get_shape().as_list()
                # pdb.set_trace()
                imgSrcDom = tf.tile(tf.expand_dims(imgSrcDom, axis=-1), [1,1,1,1,options.segment_class])
                imgSrcDom = tf.reshape(tf.transpose(imgSrcDom,[0,1,2,4,3]), [b,h,w,-1])
                maskSrcDom = tf.tile(tf.expand_dims(maskSrcDom, axis=-1), [1,1,1,1,c])
                maskSrcDom = tf.reshape(maskSrcDom, [b,h,w,-1])
                x = tf.reduce_sum(tf.multiply(imgSrcDom, maskSrcDom), axis=-1, keep_dims=True)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, scope='conv_src')
            elif (imgSrcDom is None) and (maskSrcDom is None) and (imgDstDom is not None):
                ## Not utilize the seg label
                input_channel  = imgDstDom.get_shape().as_list()[-1]
                x = imgDstDom
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, scope='conv_dst')
            else:
                raise Exception('Input is not right !')

        with tf.variable_scope('weight_sharing_2', reuse=reuse2) as sc:
            # Encoder
            encoder_layer_list = []
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                # channel_num = x.get_shape()[-1]
                res = x
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                encoder_layer_list.append(x)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn)
            
            for idx in range(repeat_num):
                x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
                res = x
                # channel_num = hidden_num * (repeat_num-idx)
                channel_num = x.get_shape()[-1]
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn)
                x = x + res
                if idx < repeat_num - 1:
                    # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn)
                    x = upscale(x, 2)
                    x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn)

            # pdb.set_trace()
            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None)

    # variables = tf.contrib.framework.get_variables(vs)
    return out

def generator_UAEAfterResidual_PG2_imgseg_diffMap(imgSrcDom, maskSrcDom, imgDstDom, options, reuse1=False, reuse2=False, use_deconv=False, repeat_num=5, activation_fn=tf.nn.elu, name="generator"):
    diffMap = generator_UAEAfterResidual_PG2_imgseg(imgSrcDom, maskSrcDom, imgDstDom, options, reuse1, reuse2, use_deconv, repeat_num, activation_fn, name)
    if (imgSrcDom is not None) and (maskSrcDom is not None) and (imgDstDom is None):
        out = imgSrcDom + diffMap
    elif (imgSrcDom is None) and (maskSrcDom is None) and (imgDstDom is not None):
        out = imgDstDom + diffMap
    else:
        raise Exception('Input is not right !')
    # variables = tf.contrib.framework.get_variables(vs)
    return out


def discriminator_imgseg(imgSrcDom, maskSrcDom, imgDstDom, options, reuse1=False, reuse2=False, name="discriminator"):
    if (imgSrcDom is not None) and (maskSrcDom is not None) and (imgDstDom is None):
        MASK_DIS = True
    elif (imgSrcDom is None) and (maskSrcDom is None) and (imgDstDom is not None):
        MASK_DIS = False
    else:
        raise Exception('Input is not right !')

    if MASK_DIS:
        x = imgSrcDom
    else:
        x = imgDstDom
    with tf.variable_scope(name) as sc:
        with tf.variable_scope('weight_sharing_1', reuse=reuse1) as sc:
            # image is 256 x 512 x input_c_dim
            h0 = lrelu(conv2d(x, options.df_dim, name='d_h0_conv'))
            # h0 is (128 x 256 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 128 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 64 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))

        with tf.variable_scope('weight_sharing_2', reuse=reuse2) as sc:
            if MASK_DIS:
                # h3 is (32 x 64 x self.df_dim*8)
                h4 = conv2d(h3, options.segment_class, s=1, name='d_h4_conv_for_mask')
                h4 = tf.reduce_sum(tf.multiply(h4, resize_like(maskSrcDom, h4)), axis=-1, keep_dims=True)
            else:
                # h3 is (32 x 64 x self.df_dim*8)
                ## use tf.tile to keep the output shape is the same
                h4 = tf.tile(conv2d(h3, 1, s=1, name='d_h4_conv'), [1,1,1,options.segment_class]) 
    return h4


def upscale(x, scale):
    _, h, w, _ = x.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(x, (h*scale, w*scale))



def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def gradloss_criterion(in_, target, weight):
    abs_deriv = tf.abs(tf.abs(tf_deriv(in_)) - tf.abs(tf_deriv(target)))
    abs_deriv = tf.reduce_mean(abs_deriv, axis=-1, keep_dims=True)
    return tf.reduce_mean(tf.multiply(weight, abs_deriv))



#################### DispNet for depth estimation ####################
from tensorflow.contrib.layers.python.layers import utils

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def disp_net(tgt_image, reuse=None):
    # Range of disparity/inverse depth values
    # DISP_SCALING = 10
    # MIN_DISP = 0.01  ## avoid NAN in log loss
    # ## value ref to https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
    # DISP_SCALING = 0.3
    # MIN_DISP = 0.
    DISP_SCALING = 1.
    MIN_DISP = 0.
    
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4]
            # return disp1


#################### Vgg/Res50 for Disparity ####################
## ref to https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py
def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def conv_nn(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
    conv1 = conv_nn(x,     num_out_layers, kernel_size, 1)
    conv2 = conv_nn(conv1, num_out_layers, kernel_size, 2)
    return conv2

def maxpool_nn(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def resconv_nn(x, num_layers, stride):
    do_proj = tf.shape(x)[3] != num_layers or stride == 2
    shortcut = []
    conv1 = conv_nn(x,         num_layers, 1, 1)
    conv2 = conv_nn(conv1,     num_layers, 3, stride)
    conv3 = conv_nn(conv2, 4 * num_layers, 1, 1, None)
    if do_proj:
        shortcut = conv_nn(x, 4 * num_layers, 1, stride, None)
    else:
        shortcut = x
    return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks, down_sample=True):
    out = x
    for i in range(num_blocks - 1):
        out = resconv_nn(out, num_layers, 1)
    if down_sample:
        out = resconv_nn(out, num_layers, 2)
    else:
        out = resconv_nn(out, num_layers, 1)
    return out

def upconv_nn(x, num_out_layers, kernel_size, scale):
    upsample = upsample_nn(x, scale)
    conv = conv_nn(upsample, num_out_layers, kernel_size, 1)
    return conv

def deconv_nn(x, num_out_layers, kernel_size, scale):
    p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
    return conv[:,3:-1,3:-1,:]

def get_disp(x):
    # disp = 0.3 * conv_nn(x, 2, 3, 1, tf.nn.sigmoid) ## for left + right camera
    disp = 1.0 * conv_nn(x, 1, 3, 1, tf.nn.sigmoid) + 0.0 ## for monocular camera
    # disp = 1.0 * conv_nn(x, 1, 3, 1, None) ## for monocular camera
    return disp

def get_seg(x, class_num):
    seg = conv_nn(x, class_num, 3, 1, None)
    return seg

def disp_vgg(model_input, use_deconv=False, reuse=None):
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_block(model_input,  32, 7) # H/2
            conv2 = conv_block(conv1,             64, 5) # H/4
            conv3 = conv_block(conv2,            128, 3) # H/8
            conv4 = conv_block(conv3,            256, 3) # H/16
            conv5 = conv_block(conv4,            512, 3) # H/32
            conv6 = conv_block(conv5,            512, 3) # H/64
            conv7 = conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv_func(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv_nn(concat7,  512, 3, 1)

            upconv6 = upconv_func(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv_nn(concat6,  512, 3, 1)

            upconv5 = upconv_func(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv_nn(concat5,  256, 3, 1)

            upconv4 = upconv_func(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv_nn(concat4,  128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv_func(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv_nn(concat3,   64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv_func(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv_nn(concat2,   32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv_func(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv_nn(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)
        
    return [disp1, disp2, disp3, disp4]

def disp_resnet50(model_input, use_deconv=False, reuse=None):
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv_func(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv_nn(concat6,   512, 3, 1)

            upconv5 = upconv_func(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv_nn(concat5,   256, 3, 1)

            upconv4 = upconv_func(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv_nn(concat4,   128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv_func(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv_nn(concat3,    64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv_func(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv_nn(concat2,    32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv_func(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv_nn(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)
            # pdb.set_trace()
            # disp1=disp1
        
    return [disp1, disp2, disp3, disp4]

def seg_disp_resnet50(model_input, class_num, use_deconv=False, reuse=None):
    with tf.variable_scope('seg_depth_net', reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv_func(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv_nn(concat6,   512, 3, 1)

            upconv5 = upconv_func(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv_nn(concat5,   256, 3, 1)

            upconv4 = upconv_func(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv_nn(concat4,   128, 3, 1)
            seg4 = get_seg(iconv4, class_num)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv_func(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv_nn(concat3,    64, 3, 1)
            seg3 = get_seg(iconv3, class_num)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv_func(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv_nn(concat2,    32, 3, 1)
            seg2 = get_seg(iconv2, class_num)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv_func(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv_nn(concat1,   16, 3, 1)
            seg1 = get_seg(iconv1, class_num)
            disp1 = get_disp(iconv1)
            # pdb.set_trace()
            # disp1=disp1
        
    return [seg1, seg2, seg3, seg4], [disp1, disp2, disp3, disp4]

def seg_disp_branch_resnet50(model_input, class_num, use_deconv=False, reuse=None, name='seg_depth_net'):
    with tf.variable_scope(name, reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D
            common_fea = conv5

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):

            with tf.variable_scope('seg_branch'):
                upconv6_seg = upconv_func(conv5,   512, 3, 2) #H/32
                concat6_seg = tf.concat([upconv6_seg, skip5], 3)
                iconv6_seg  = conv_nn(concat6_seg,   512, 3, 1)

                upconv5_seg = upconv_func(iconv6_seg, 256, 3, 2) #H/16
                concat5_seg = tf.concat([upconv5_seg, skip4], 3)
                iconv5_seg  = conv_nn(concat5_seg,   256, 3, 1)

                upconv4_seg = upconv_func(iconv5_seg,  128, 3, 2) #H/8
                concat4_seg = tf.concat([upconv4_seg, skip3], 3)
                iconv4_seg  = conv_nn(concat4_seg,   128, 3, 1)
                seg4 = get_seg(iconv4_seg, class_num)
                useg4  = upsample_nn(seg4, 2)

                upconv3_seg = upconv_func(iconv4_seg,   64, 3, 2) #H/4
                concat3_seg = tf.concat([upconv3_seg, skip2, useg4], 3)
                iconv3_seg  = conv_nn(concat3_seg,    64, 3, 1)
                seg3 = get_seg(iconv3_seg, class_num)
                useg3  = upsample_nn(seg3, 2)

                upconv2_seg = upconv_func(iconv3_seg,   32, 3, 2) #H/2
                concat2_seg = tf.concat([upconv2_seg, skip1, useg3], 3)
                iconv2_seg  = conv_nn(concat2_seg,    32, 3, 1)
                seg2 = get_seg(iconv2_seg, class_num)
                useg2  = upsample_nn(seg2, 2)

                upconv1_seg = upconv_func(iconv2_seg,  16, 3, 2) #H
                concat1_seg = tf.concat([upconv1_seg, useg2], 3)
                iconv1_seg  = conv_nn(concat1_seg,   16, 3, 1)
                seg1 = get_seg(iconv1_seg, class_num)

                seg_fea = iconv1_seg

            with tf.variable_scope('disp_branch'):
                upconv6_disp = upconv_func(conv5,   512, 3, 2) #H/32
                concat6_disp = tf.concat([upconv6_disp, skip5], 3)
                iconv6_disp  = conv_nn(concat6_disp,   512, 3, 1)

                upconv5_disp = upconv_func(iconv6_disp, 256, 3, 2) #H/16
                concat5_disp = tf.concat([upconv5_disp, skip4], 3)
                iconv5_disp  = conv_nn(concat5_disp,   256, 3, 1)

                upconv4_disp = upconv_func(iconv5_disp,  128, 3, 2) #H/8
                concat4_disp = tf.concat([upconv4_disp, skip3], 3)
                iconv4_disp  = conv_nn(concat4_disp,   128, 3, 1)
                disp4 = get_disp(iconv4_disp)
                udisp4  = upsample_nn(disp4, 2)

                upconv3_disp = upconv_func(iconv4_disp,   64, 3, 2) #H/4
                concat3_disp = tf.concat([upconv3_disp, skip2, udisp4], 3)
                iconv3_disp  = conv_nn(concat3_disp,    64, 3, 1)
                disp3 = get_disp(iconv3_disp)
                udisp3  = upsample_nn(disp3, 2)

                upconv2_disp = upconv_func(iconv3_disp,   32, 3, 2) #H/2
                concat2_disp = tf.concat([upconv2_disp, skip1, udisp3], 3)
                iconv2_disp  = conv_nn(concat2_disp,    32, 3, 1)
                disp2 = get_disp(iconv2_disp)
                udisp2  = upsample_nn(disp2, 2)

                upconv1_disp = upconv_func(iconv2_disp,  16, 3, 2) #H
                concat1_disp = tf.concat([upconv1_disp, udisp2], 3)
                iconv1_disp  = conv_nn(concat1_disp,   16, 3, 1)
                disp1 = get_disp(iconv1_disp)

                disp_fea = iconv1_disp
        
    return [seg1, seg2, seg3, seg4], [disp1, disp2, disp3, disp4], common_fea, seg_fea, disp_fea

def disp_resnet50(model_input, use_deconv=False, reuse=None):
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv_func(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv_nn(concat6,   512, 3, 1)

            upconv5 = upconv_func(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv_nn(concat5,   256, 3, 1)

            upconv4 = upconv_func(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv_nn(concat4,   128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv_func(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv_nn(concat3,    64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv_func(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv_nn(concat2,    32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv_func(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv_nn(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)
            # pdb.set_trace()
            # disp1=disp1
        
    return [disp1, disp2, disp3, disp4]

def seg_branch_resnet50(model_input, class_num, use_deconv=False, reuse=None, name='seg_net'):
    with tf.variable_scope(name, reuse=reuse) as sc:
        #set convenience functions
        if use_deconv:
            upconv_func = deconv_nn
        else:
            upconv_func = upconv_nn

        with tf.variable_scope('encoder'):
            conv1 = conv_nn(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool_nn(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D
            common_fea = conv5

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
        
        # DECODING
        with tf.variable_scope('decoder'):

            with tf.variable_scope('seg_branch'):
                upconv6_seg = upconv_func(conv5,   512, 3, 2) #H/32
                concat6_seg = tf.concat([upconv6_seg, skip5], 3)
                iconv6_seg  = conv_nn(concat6_seg,   512, 3, 1)

                upconv5_seg = upconv_func(iconv6_seg, 256, 3, 2) #H/16
                concat5_seg = tf.concat([upconv5_seg, skip4], 3)
                iconv5_seg  = conv_nn(concat5_seg,   256, 3, 1)

                upconv4_seg = upconv_func(iconv5_seg,  128, 3, 2) #H/8
                concat4_seg = tf.concat([upconv4_seg, skip3], 3)
                iconv4_seg  = conv_nn(concat4_seg,   128, 3, 1)
                seg4 = get_seg(iconv4_seg, class_num)
                useg4  = upsample_nn(seg4, 2)

                upconv3_seg = upconv_func(iconv4_seg,   64, 3, 2) #H/4
                concat3_seg = tf.concat([upconv3_seg, skip2, useg4], 3)
                iconv3_seg  = conv_nn(concat3_seg,    64, 3, 1)
                seg3 = get_seg(iconv3_seg, class_num)
                useg3  = upsample_nn(seg3, 2)

                upconv2_seg = upconv_func(iconv3_seg,   32, 3, 2) #H/2
                concat2_seg = tf.concat([upconv2_seg, skip1, useg3], 3)
                iconv2_seg  = conv_nn(concat2_seg,    32, 3, 1)
                seg2 = get_seg(iconv2_seg, class_num)
                useg2  = upsample_nn(seg2, 2)

                upconv1_seg = upconv_func(iconv2_seg,  16, 3, 2) #H
                concat1_seg = tf.concat([upconv1_seg, useg2], 3)
                iconv1_seg  = conv_nn(concat1_seg,   16, 3, 1)
                seg1 = get_seg(iconv1_seg, class_num)

                seg_fea = iconv1_seg

    return [seg1, seg2, seg3, seg4], common_fea, seg_fea


'''
============================================================================
LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
============================================================================
Based on the paper: https://arxiv.org/pdf/1707.03718.pdf
'''
#TODO: net initialization
from tensorflow.contrib.layers.python.layers import initializers
def LinkNet_arg_scope(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  '''
  The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
  Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope



#Now actually start building the network
def LinkNet(inputs,
         num_classes,
         reuse=None,
         is_training=True,
         feature_scale=4,
         scope='LinkNet'):
    '''
    The ENet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    @slim.add_arg_scope
    def convBnRelu(x, num_channel, kernel_size, stride, is_training, scope, padding = 'SAME'):
        x = slim.conv2d(x, num_channel, [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_conv1', padding = padding)
        x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
        x = tf.nn.relu(x, name=scope+'_relu1')
        return x


    @slim.add_arg_scope
    def deconvBnRelu(x, num_channel, kernel_size, stride, is_training, scope, padding = 'VALID'):
        # pdb.set_trace()
        x = slim.conv2d_transpose(x, int(num_channel), [kernel_size, kernel_size], stride=stride, activation_fn=None, scope=scope+'_fullconv1', padding = padding)
        x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm1')
        x = tf.nn.relu(x, name=scope+'_relu1')
        return x    

    @slim.add_arg_scope
    def initial_block(inputs, is_training=True, scope='initial_block'):
        '''
        The initial block for Linknet has 2 branches: The convolution branch and Maxpool branch.
        INPUTS:
        - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
        OUTPUTS:
        - net_concatenated(Tensor): a 4D Tensor that contains the 
        '''
        #Convolutional branch
        net_conv = slim.conv2d(inputs, 64, [7,7], stride=2, activation_fn=None, scope=scope+'_conv')
        net_conv = slim.batch_norm(net_conv, is_training=is_training, fused=True, scope=scope+'_batchnorm')
        net_conv = tf.nn.relu(net_conv, name=scope+'_relu')

        #Max pool branch
        net_pool = slim.max_pool2d(net_conv, [3,3], stride=2, scope=scope+'_max_pool')
        return net_conv

    @slim.add_arg_scope
    def residualBlock(x, n_filters, is_training, stride=1, downsample=None, scope='residualBlock'):
        # Shortcut connection
        # Downsample the data or just pass original
        if downsample == None:
            shortcut = x
        else:
            shortcut = downsample

        # Residual
        x = convBnRelu(x, n_filters, kernel_size = 3, stride = stride, is_training = is_training, scope = scope + '/cvbnrelu')
        x = slim.conv2d(x, n_filters, [3,3], stride=1, activation_fn=None, scope=scope+'_conv2',  padding = 'SAME')
        x = slim.batch_norm(x, is_training=is_training, fused=True, scope=scope+'_batchnorm2')
        
        # Shortcutr connection
        x = x + shortcut
        x = tf.nn.relu(x, name=scope+'_relu2')
        return x


    @slim.add_arg_scope
    def encoder(inputs, inplanes, planes, blocks, stride, is_training=True, scope='encoder'):
        '''
        Decoder of LinkNet
        INPUTS:
        - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
        OUTPUTS:
        - net_concatenated(Tensor): a 4D Tensor that contains the 
        '''        
        # make downsample at skip connection if needed
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample   = slim.conv2d(inputs, planes, [1,1], stride=stride, activation_fn=None, scope=scope+'_conv_downsample')
            downsample   = slim.batch_norm(downsample, is_training=is_training, fused=True, scope=scope+'_batchnorm_downsample')

        # Create mupliple block of ResNet
        output = residualBlock(inputs, planes, is_training, stride, downsample, scope = scope +'/residualBlock0')
        for i in range(1, blocks):
            output = residualBlock(output, planes, is_training, 1, scope = scope +'/residualBlock{}'.format(i))
        return output

    @slim.add_arg_scope
    def decoder(inputs, n_filters, planes, is_training=True, scope='decoder'):
        '''
        Encoder use ResNet block. As in paper, we  will use ResNet18 block for learning.
        INPUTS:
        - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
        OUTPUTS:
        - net_concatenated(Tensor): a 4D Tensor that contains the 
        '''        
        # pdb.set_trace()
        x = convBnRelu(inputs, int(n_filters/2), kernel_size = 1, stride = 1, is_training = is_training,  padding = 'SAME', scope = scope + "/c1")
        x = deconvBnRelu(x, int(n_filters/2), kernel_size = 3, stride = 2, is_training = is_training, padding = 'SAME', scope = scope+ "/dc1")
        x = convBnRelu(x, planes, kernel_size = 1, stride = 1, is_training = is_training,  padding = 'SAME', scope = scope+ "/c2")
        return  x    


    #Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    # inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    layers = [2, 2, 2, 2]
    filters = [64, 128, 256, 512]
    filters = [int(x / feature_scale) for x in filters]

    with tf.variable_scope(scope, reuse=reuse):
        #Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([initial_block, encoder], is_training=is_training),\
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None): 
            #=================INITIAL BLOCK=================
           
            net = initial_block(inputs, scope='initial_block')

            #===================Encoder=======================
            enc1 = encoder(net,  64, filters[0], layers[0], stride=1, is_training=is_training, scope='encoder1')
            enc2 = encoder(enc1, filters[0], filters[1], layers[1], stride=2, is_training=is_training, scope='encoder2')
            enc3 = encoder(enc2, filters[1], filters[2], layers[2], stride=2, is_training=is_training, scope='encoder3')
            enc4 = encoder(enc3, filters[2], filters[3], layers[3], stride=2, is_training=is_training, scope='encoder4')

           
            #===================Decoder=======================
            decoder4 = decoder(enc4, filters[3], filters[2], is_training=is_training, scope='decoder4')
            decoder4 += enc3
            decoder3 = decoder(decoder4, filters[2], filters[1], is_training=is_training, scope='decoder3')
            decoder3 += enc2
            decoder2 = decoder(decoder3, filters[1], filters[0], is_training=is_training, scope='decoder2')
            decoder2 += enc1
            decoder1 = decoder(decoder2, filters[0], filters[0], is_training=is_training, scope='decoder1')

            #===================Final Classification=======================
            f1     = deconvBnRelu(decoder1, 32/feature_scale, 3, stride = 2, is_training=is_training, scope='f1',padding = 'SAME')
            f2     = convBnRelu(f1, 32/feature_scale, 3, stride = 1, is_training=is_training, padding = 'SAME', scope='f2')
            logits = slim.conv2d(f2, num_classes, [2,2], stride=2, activation_fn=None, padding = 'SAME', scope='logits')

        return logits


## Gaussian blur
## ref to https://github.com/antonilo/TensBlur/blob/master/smoother.py
import scipy.stats as st
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Smoother(object):
    def __init__(self, inputs, filter_size, sigma):
        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.filter_size = filter_size
        self.sigma = sigma
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(name = 'smoothing'))

    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter

    def make_gauss_var(self, name, size, sigma, c_i):
        with tf.device("/cpu:0"):
            kernel = self.gauss_kernel(size, sigma, c_i)
            var = tf.Variable(tf.convert_to_tensor(kernel), name = name)
        return var

    def get_output(self):
        '''Returns the smoother output.'''
        return self.terminals[-1]

    @layer
    def conv(self,
             input,
             name,
             padding='SAME'):
        # Get the number of channels in the input
        c_i = input.get_shape().as_list()[3]
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                             padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_gauss_var('gauss_weight', self.filter_size,
                                                         self.sigma, c_i)
            output = convolve(input, kernel)
            return output
