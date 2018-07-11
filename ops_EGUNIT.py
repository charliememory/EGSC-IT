import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import variance_scaling_initializer as he_init

PAD_MODE = 'REFLECT' # 'CONSTANT' or 'REFLECT'


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)
        
def apply_ins_norm_2d(x, gamma, beta):
    assert len(x.get_shape().as_list())==4
    _, h, w, _ = x.get_shape().as_list()
    mean, var = tf.nn.moments(x, [1,2])
    mean = tf.tile(tf.expand_dims(tf.expand_dims(mean, 1), 1), [1,h,w,1])
    var = tf.tile(tf.expand_dims(tf.expand_dims(var, 1), 1), [1,h,w,1])
    gamma2 = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 1), 1), [1,h,w,1])
    beta2 = tf.tile(tf.expand_dims(tf.expand_dims(beta, 1), 1), [1,h,w,1])
    x = (1+gamma2)*((x-mean)/var) + beta2
    # x = gamma2*((x-mean)/var) + beta2
    return x

def apply_ins_norm_2d_like(x, ref):
    assert len(x.get_shape().as_list())==4
    assert len(ref.get_shape().as_list())==4    
    gamma, beta = tf.nn.moments(ref, [1,2])
    _, h, w, _ = x.get_shape().as_list()
    mean, var = tf.nn.moments(x, [1,2])
    mean = tf.tile(tf.expand_dims(tf.expand_dims(mean, 1), 1), [1,h,w,1])
    var = tf.tile(tf.expand_dims(tf.expand_dims(var, 1), 1), [1,h,w,1])
    gamma2 = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 1), 1), [1,h,w,1])
    beta2 = tf.tile(tf.expand_dims(tf.expand_dims(beta, 1), 1), [1,h,w,1])
    x = (gamma2)*((x-mean)/var) + beta2
    return x, gamma, beta

def fc(x, channels, normal_weight_init=False, activation_fn='leaky', scope='fc_0') :
    with tf.variable_scope(scope) :
        x = tf.layers.flatten(x)

        if normal_weight_init :
            x = tf.layers.dense(inputs=x, units=channels, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        else :
            if activation_fn == 'relu' :
                x = tf.layers.dense(inputs=x, units=channels, kernel_initializer=he_init(),
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.dense(inputs=x, units=channels, kernel_size=kernel, 
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))


        x = activation(x, activation_fn)

        return x

def conv(x, channels, kernel=3, stride=2, pad=0, normal_weight_init=False, activation_fn='leaky', is_training=True, norm_fn=None, scope='conv_0') :
    with tf.variable_scope(scope) :
        x = tf.pad(x, [[0,0], [pad, pad], [pad, pad], [0,0]], mode=PAD_MODE)

        if normal_weight_init :
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        else :
            if activation_fn == 'relu' :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        if norm_fn == 'instance' :
            x = instance_norm(x, 'ins_norm')
        if norm_fn == 'batch' :
            x = batch_norm(x, is_training, 'batch_norm')

        x = activation(x, activation_fn)

        return x

def deconv(x, channels, kernel=3, stride=2, normal_weight_init=False, activation_fn='leaky', scope='deconv_0') :
    with tf.variable_scope(scope):
        if normal_weight_init:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, padding='SAME', kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        else:
            if activation_fn == 'relu' :
                x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride, padding='SAME',
                                               kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, strides=stride, padding='SAME',
                                               kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        x = activation(x, activation_fn)

        return x

def resblock(x_init, channels, kernel=3, stride=1, pad=1, dropout_ratio=0.0, normal_weight_init=False, is_training=True, norm_fn='instance', scope='resblock_0') :
    assert norm_fn in ['instance', 'batch', 'weight', 'spectral', None]
    with tf.variable_scope(scope) :
        with tf.variable_scope('res1') :
            x = tf.pad(x_init, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=PAD_MODE)

            if normal_weight_init :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

            if norm_fn == 'instance' :
                x = instance_norm(x, 'res1_instance')
            if norm_fn == 'batch' :
                x = batch_norm(x, is_training, 'res1_batch')

            x = relu(x)
        with tf.variable_scope('res2') :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=PAD_MODE)

            if normal_weight_init :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

            if norm_fn == 'instance' :
                x = instance_norm(x, 'res2_instance')
            if norm_fn == 'batch' :
                x = batch_norm(x, is_training, 'res2_batch')

        if dropout_ratio > 0.0 :
            x = tf.layers.dropout(x, rate=dropout_ratio, training=is_training)

        return x + x_init

def activation(x, activation_fn='leaky') :
    assert activation_fn in ['relu', 'leaky', 'tanh', 'sigmoid', 'swish', None]
    if activation_fn == 'leaky':
        x = lrelu(x)

    if activation_fn == 'relu':
        x = relu(x)

    if activation_fn == 'sigmoid':
        x = sigmoid(x)

    if activation_fn == 'tanh' :
        x = tanh(x)

    if activation_fn == 'swish' :
        x = swish(x)

    return x

def lrelu(x, alpha=0.2) :
    # UNIT pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def tanh(x) :
    return tf.tanh(x)

def swish(x) :
    return x * sigmoid(x)

def batch_norm(x, is_training=False, scope='batch_nom') :
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def instance_norm(x, scope='instance', trainable=True, reuse=None) :
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope,
                                           trainable=trainable,
                                           reuse=reuse)

def adaptive_BN(x, is_training=False):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*batch_norm(x, is_training) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def adaptive_IN(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*instance_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def gaussian_noise_layer(mu):
    sigma = 1.0
    gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mu + sigma * gaussian_random_vector

## TODO
def GaussianVAE2D(x, scope='vae2d'):
    mu = conv(x, x.get_shape().as_list()[-1], kernel=1, stride=1, pad=0, normal_weight_init=True, activation_fn='leaky', scope='conv_mu')
    # sigma = 1.0
    sigma = conv(x, x.get_shape().as_list()[-1], kernel=1, stride=1, pad=0, normal_weight_init=True, activation_fn='leaky', scope='conv_sigma')
    sigma = tf.nn.softplus(sigma)
    gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mu+sigma*gaussian_random_vector, mu, sigma
    
def KLD_mu_sd(mu, sigma) :
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    loss = tf.reduce_mean(KL_divergence)
    # mu_2 = tf.square(mu)
    # loss = tf.reduce_mean(mu_2)

    return loss

def KL_divergence(mu) :
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)

    return loss

def L1_loss(x, y) :
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def L2_loss(x, y) :
    loss = 0.5*tf.reduce_mean(tf.square(x - y))
    return loss

def discriminator_loss(real, fake, smoothing=False, use_lasgan=False) :
    if use_lasgan :
        if smoothing :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 0.9))
        else :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))

        fake_loss = tf.reduce_mean(tf.square(fake))
    else :
        if smoothing :
            real_labels = tf.fill(tf.shape(real), 0.9)
        else :
            real_labels = tf.ones_like(real)

        fake_labels = tf.zeros_like(fake)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    loss = (real_loss + fake_loss) * 0.5

    return loss

def generator_loss(fake, smoothing=False, use_lsgan=False) :
    if use_lsgan :
        if smoothing :
            loss = tf.reduce_mean(tf.squared_difference(fake, 0.9))
        else :
            loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))
    else :
        if smoothing :
            fake_labels = tf.fill(tf.shape(fake), 0.9)
        else :
            fake_labels = tf.ones_like(fake)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    return loss

## ref https://github.com/tensorflow/magenta/blob/master/magenta/models/image_stylization/learning.py
def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps."""
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
      feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator

# def l2_norm(f, axis=None, keep_dims=False):
#     return tf.sqrt(tf.reduce_sum(tf.square(f), axis=axis, keepdims=keep_dims))


# def l2_normalise(v, axis):
#     return v / l2_norm(v, axis=axis, keep_dims=True)



## ref to https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/master/tests/UnpoolLayerTest.ipynb
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

def unpool_layer2x2(x, raveled_argmax, out_shape):
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    
    # pdb.set_trace()
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

## ref https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/91d1d56df5b966454f3c24988e380ffb75184f02/DeconvNetPipeline.py#L241
def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = bottom.get_shape().as_list()
    top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])
    return tf.scatter_nd(indices, values, tf.to_int64(top_shape))


## ref https://github.com/rayanelleuch/tensorflow/blob/b46d50583d8f4893f1b1d629d0ac9cb2cff580af/tensorflow/contrib/layers/python/layers/layers.py#L2291-L2327
## ref https://github.com/tensorflow/tensorflow/pull/16885/commits/de11499062c33aeac9fd901d6b07a33a1eb9cb83
# @add_arg_scope
def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
  """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366
  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
  """
  with tf.variable_scope(scope):
    input_shape = tf.shape(pool)
    output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

    flat_input_size = tf.reduce_prod(input_shape)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                      shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b1 = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b1, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
    ret = tf.reshape(ret, output_shape)

    set_input_shape = pool.get_shape()
    set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
    ret.set_shape(set_output_shape)
    return ret




## ref https://raw.githubusercontent.com/antonilo/TensBlur/master/smoother.py
## Gaussian blur
import scipy.stats as st
import numpy as np
import pdb
def conv_gaussian_blur(input, filter_size=3, sigma=1, name='gaussian_blur', padding='SAME'):
    def gauss_kernel(kernlen=21, nsig=3, channels=1):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter

    def make_gauss_var(name, size, sigma, c_i):
        # with tf.device("/cpu:0"):
        kernel = gauss_kernel(size, sigma, c_i)
        # pdb.set_trace()
        var = tf.Variable(tf.convert_to_tensor(kernel), name = name)
        return var
        
    # Get the number of channels in the input
    c_i = input.get_shape().as_list()[3]
    # Convolution for a given input and kernel
    convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                         padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_gauss_var('gauss_weight', filter_size, sigma, c_i)
        output = convolve(input, kernel)
        return output

