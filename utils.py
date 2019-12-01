from __future__ import division
import math
import pprint
import scipy.misc
import scipy.ndimage
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import reduce

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

#### Count param number
def count_params():
    "print number of trainable variables"
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    n = sum(size(v) for v in tf.trainable_variables())
    print("Model size: {}K".format(n/1000,))

#######################################################################
############################ I/O functions ############################
#######################################################################
class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            tmp3 = copy.copy(self.images[idx])[2]
            self.images[idx][0] = image[0]
            self.images[idx][2] = image[2]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            tmp4 = copy.copy(self.images[idx])[3]
            self.images[idx][1] = image[1]
            self.images[idx][3] = image[3]
            return [tmp1, tmp2, tmp3, tmp4]
        else:
            return image

class ImagePool_UNIT:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
def load_test_data(image_path, image_width=512, image_height=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [image_height, image_width])
    img = img/127.5 - 1
    return img

def one_hot(image_in, num_classes=8):
  hot = np.zeros((image_in.shape[0], image_in.shape[1], num_classes))
  layer_idx = np.arange(image_in.shape[0]).reshape(image_in.shape[0], 1)
  component_idx = np.tile(np.arange(image_in.shape[1]), (image_in.shape[0], 1))
  hot[layer_idx, component_idx, image_in] = 1
  return hot.astype(np.int)

def load_train_data(image_path, image_width=512, image_height=256, num_seg_masks=8, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    seg_A = imread(image_path[0].replace("trainA","trainA_seg"))
    seg_class_A = scipy.misc.imread(image_path[0].replace("trainA","trainA_seg_class")) if not is_testing else None
    seg_B = imread(image_path[1].replace("trainB","trainB_seg"))
    seg_class_B = scipy.misc.imread(image_path[1].replace("trainB","trainB_seg_class")) if not is_testing else None
    # preprocess seg masks
    if not is_testing:
        seg_mask_A = one_hot(seg_class_A.astype(np.int), num_seg_masks)
        seg_mask_B = one_hot(seg_class_B.astype(np.int), num_seg_masks)
    else:
        seg_mask_A = None
        seg_mask_B = None

    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [image_height, image_width])
        seg_A = scipy.misc.imresize(seg_A, [image_height, image_width])
        seg_mask_A = scipy.ndimage.interpolation.zoom(seg_mask_A, (image_height/8.0/seg_mask_A.shape[0], image_width/8.0/seg_mask_A.shape[1],1), mode="nearest")
        
        img_B = scipy.misc.imresize(img_B, [image_height, image_width])
        seg_B = scipy.misc.imresize(seg_B, [image_height, image_width])
        seg_mask_B = scipy.ndimage.interpolation.zoom(seg_mask_B, (image_height/8.0/seg_mask_B.shape[0], image_width/8.0/seg_mask_B.shape[1],1), mode="nearest")

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
            seg_A = np.fliplr(seg_A)
            seg_B = np.fliplr(seg_B)
            seg_mask_A = np.fliplr(seg_mask_A)
            seg_mask_B = np.fliplr(seg_mask_B)
    else:
        img_A = scipy.misc.imresize(img_A, [image_height, image_width])
        img_B = scipy.misc.imresize(img_B, [image_height, image_width])
        seg_A = scipy.misc.imresize(seg_A, [image_height, image_width])
        seg_B = scipy.misc.imresize(seg_B, [image_height, image_width])
 
    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.
    seg_A = seg_A/127.5 - 1.
    seg_B = seg_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    seg_AB = np.concatenate((seg_A, seg_B), axis=2)
    # img_AB shape: (image_height, image_width, input_c_dim + output_c_dim)
    return img_AB, seg_AB, seg_mask_A, seg_mask_B

# -----------------------------

#def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
#    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.zeros((h * int(size[0]), w * int(size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


#######################################################################
######################### Transform functions #########################
#######################################################################
def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def process_image(image, mean_pixel, norm):
    return (image - mean_pixel) / norm

def unprocess_image(image, mean_pixel, norm):
    return image * norm + mean_pixel


import tensorflow as tf
import pdb
def data_augment(rgb, # 3 channels
                seg=None, # 3 channels
                seg_class_map=None, # seg_class channels
                depth_mat=None, # 1 channel
                flow_mat=None, # 1 channel
                resize=None, # [width, height] list or None
                horizontal_flip=False,
                vertical_flip=False,
                rotate=0, # Maximum rotation angle in degrees
                crop_probability=0, # How often we do crops
                color_probability=0): # How often we do color jitter
                # mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
    ## ref https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
    ## ref https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image
    ## ref https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
    assert not ((depth_mat is not None) and (rotate!=0)), 'depth can not be rotated directly!'
    # assert not ((depth_mat is not None) and (crop_probability>0)), 'depth can not be cropped directly!'

    if rgb.dtype != tf.float32:
        rgb = process_image(tf.to_float(rgb), 127.5, 127.5)
    if seg is not None and seg.dtype!=tf.float32:
        seg = process_image(tf.to_float(seg), 127.5, 127.5)

    if resize is not None:
        rgb = tf.image.resize_images(rgb, resize, method=tf.image.ResizeMethod.BILINEAR, align_corners=True) ## or tf.image.ResizeMethod.BILINEAR
        if seg is not None:
            seg = tf.image.resize_images(seg, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        if seg_class_map is not None:
            seg_class_map = tf.image.resize_images(seg_class_map, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        if depth_mat is not None:
            depth_mat = tf.image.resize_images(depth_mat, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
            
    with tf.name_scope('augmentation'):
        shp = tf.shape(rgb)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

        if crop_probability > 0:
            min_crop_percent = 0.8 # Minimum linear dimension of a crop for left-top/right-bottom
            max_crop_percent = 1. # Maximum linear dimension of a crop for left-top/right-bottom

            crop_pct = tf.random_uniform([batch_size], min_crop_percent,
                                       max_crop_percent)
            ## ref to paper "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
            if depth_mat is not None: # Depth map can only be center-scaled and Need to modify depth value with: depth *= crop_pct
                crop_pct_expand = tf.expand_dims(tf.expand_dims(tf.expand_dims(crop_pct,axis=-1),axis=-1),axis=-1)
                depth_mat = tf.multiply(depth_mat, tf.tile(crop_pct_expand, [1,tf.to_int32(height),tf.to_int32(width),1])) # depth *= crop_pct
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            # top = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            # left = width * (1 - crop_pct)
            # top = height * (1 - crop_pct)
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), left,
                tf.zeros([batch_size]), crop_pct, top,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                        tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            # transforms[-1] = tf.Print(transforms[-1], [transforms[-1]], summarize=80, message="transforms[-1] is:")

        if transforms:
            rgb = tf.contrib.image.transform(
                rgb,
                tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR') # or 'NEAREST'
            if seg is not None:
                seg = tf.contrib.image.transform(
                    seg,
                    tf.contrib.image.compose_transforms(*transforms),
                        interpolation='NEAREST')
            if seg_class_map is not None:
                seg_class_map = tf.contrib.image.transform(
                    seg_class_map,
                    tf.contrib.image.compose_transforms(*transforms),
                        interpolation='NEAREST')
            if depth_mat is not None:
                depth_mat = tf.contrib.image.transform(
                    depth_mat,
                    tf.contrib.image.compose_transforms(*transforms),
                        interpolation='NEAREST')

        if color_probability:
            rgb2 = tf.image.random_brightness(rgb, max_delta=32. / 255.)
            rgb2 = tf.image.random_saturation(rgb2, lower=0.1, upper=1.1)
            rgb2 = tf.image.random_hue(rgb2, max_delta=0.1)
            rgb2 = tf.image.random_contrast(rgb2, lower=0.1, upper=1.1)
            # The random_* ops do not necessarily clamp.
            rgb2 = tf.clip_by_value(rgb2, 0.0, 1.0)
            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), color_probability)
            rgb = tf.where(coin, rgb, rgb2)

        # def cshift(values): # Circular shift in batch dimension
        #     return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

        # if mixup > 0:
        #     beta = tf.distributions.Beta(mixup, mixup)
        #     lam = beta.sample(batch_size)
        #     ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
        #     images = ll * images + (1 - ll) * cshift(images)
        #     labels = lam * labels + (1 - lam) * cshift(labels)

    return rgb, seg, seg_class_map, depth_mat, flow_mat


#######################################################################
########################### Layer functions ###########################
#######################################################################
def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs


#######################################################################
############################ Vis functions ############################
#######################################################################
def gray2heatmap(gray_img_batch, color_name='gray'):
    cmap = plt.get_cmap(color_name) 
    rgba_img_batch = cmap(np.squeeze(gray_img_batch, axis=-1))
    rgb_img_batch = np.delete(rgba_img_batch, obj=3, axis=3) ##
    return rgb_img_batch

def shrink_depth(depth_batch, thresh_max=0.5):
    depth_batch[depth_batch > thresh_max] = thresh_max ## reduce max value
    depth_batch = depth_batch/thresh_max ## norm to [0,1]
    return depth_batch


#######################################################################
######################### Evaluation functions ########################
#######################################################################
## ref to https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
def compute_depth_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

## ref to https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
#     if args.split == 'kitti':
#         gt_disp = gt_disparities[i]
#         mask = gt_disp > 0
#         pred_disp = pred_disparities_resized[i]

#         disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
#         bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
#         d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

#     abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

# print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
# print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
def eval_depth_error(gt_depth_batch, pred_depth_batch, logging, dataset='synsf'):
    batch_size, height, width, _ = gt_depth_batch.shape

    rms     = np.zeros(batch_size, np.float32)
    log_rms = np.zeros(batch_size, np.float32)
    abs_rel = np.zeros(batch_size, np.float32)
    sq_rel  = np.zeros(batch_size, np.float32)
    d1_all  = np.zeros(batch_size, np.float32)
    a1      = np.zeros(batch_size, np.float32)
    a2      = np.zeros(batch_size, np.float32)
    a3      = np.zeros(batch_size, np.float32)
    
    for i in range(batch_size):
        mask = gt_depth_batch > 0
        # if dataset == 'kitti':
        #     gt_disp = gt_disp_batch[i]
        #     mask = gt_disp > 0
        #     pred_disp = pred_disp_batch[i]

        #     disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        #     bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        #     d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_depth_errors(gt_depth_batch[mask], pred_depth_batch[mask])

    logging.info("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    logging.info("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()

## TODO
def eval_seg_error(gt_seg_batch, pred_seg_batch, dataset='synsf'):
    pass

## TODO
## ref to https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
def convert_disps_to_depths_kitti(gt_disp_batch, pred_disp_batch):
    pass

################################################################################
################################# RGB <--> LAB #################################
## ref to https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image
    
def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=-1)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return tf.stack([L_chan / 50. - 1, a_chan / 110., b_chan / 110.], axis=-1) 


def deprocess_lab(lab):
    with tf.name_scope("deprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=-1)
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2. * 100., a_chan * 110., b_chan * 110.], axis=-1)

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))


def var_filter_by_exclude(var_list, exclude_scopes=[], Print=False):    
    # exclude_scopes=['InceptionV1/Logits', 'InceptionV1/AuxLogits', 'Ver', 'Cla', 'Aux','CMC', 'Base']
    exclusions = [scope.strip() for scope in exclude_scopes]

    variables_to_restore = []
    for var in var_list:
        if Print:
            print('mlq --- variable:')
            print(var.op.name)
        excluded = False
        for exclusion in exclusions:
            if exclusion in var.op.name:
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
            if Print:
                print('restore')
        else:
            if Print:
                print('excluded')
    return variables_to_restore


