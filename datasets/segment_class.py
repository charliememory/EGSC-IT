from collections import defaultdict
import os
import sys, pdb
from glob import glob
from scipy.misc import imread, imsave, toimage
import numpy as np
from multiprocessing import Pool as ProcessPool
import argparse
from labels_utils import *


# num_seg_masks = 8
# vehicles: 1
# pedestrians: 2
# cyclist: 3
# roads: 4
# buildings: 5
# sky: 6
# tree: 7
# others: 0


# https://bitbucket.org/visinf/projects-2016-playing-for-data/src/6afee1a5923f452e741c9256f5fb78f2b3882ee2/label/initLabels.m?at=master&fileviewer=file-view-default
"""
('0,0,0', 'unlabeled')
('0,0,0', 'ego vehicle')
('0,0,0', 'rectification border')
('0,0,0', 'out of roi')
('20,20,20', 'static')
('111,74,0', 'dynamic')
('81,0,81', 'ground')
('128,64,128', 'road')
('244,35,232', 'sidewalk')
('250,170,160', 'parking')
('230,150,140', 'rail track')
('70,70,70', 'building')
('102,102,156', 'wall')
('190,153,153', 'fence')
('180,165,180', 'guard rail')
('150,100,100', 'bridge')
('150,120,90', 'tunnel')
('153,153,153', 'pole')
('153,153,153', 'polegroup')
('250,170,30', 'traffic light')
('220,220,0', 'traffic sign')
('107,142,35', 'vegetation')
('152,251,152', 'terrain')
('70,130,180', 'sky')
('220,20,60', 'person')
('255,0,0', 'rider')
('0,0,142', 'car')
('0,0,70', 'truck')
('0,60,100', 'bus')
('0,0,90', 'caravan')
('0,0,110', 'trailer')
('0,80,100', 'train')
('0,0,230', 'motorcycle')
('119,11,32', 'bicycle')
('0,0,142', 'license plate')
"""
# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def cityscape_maskmap_20trainId():
    trainId2label   = { label.color:  label.trainId for label in reversed(labels) }
    rgb_to_maskidx = defaultdict(int)
    # ## id 1: car, license plate, truck, bus, caravan, trailer, 
    # ## id 2: person, rider
    # ## id 3: motorcycle, bicycle
    # ## id 4: road, sidewalk, parking, rail track, 
    # ## id 5: building, wall, fence, guard rail, bridge, tunnel
    # ## id 6: sky
    # ## id 7: vegetation
    # ## id 0: others
    # maps = [((128,64,128),4), ((244,35,232),4), ((250,170,160),4), ((230,150,140),4), ((70,70,70),5)
    #          , ((102,102,156),5), ((190,153,153),5), ((180,165,180),5), ((150,100,100),5), ((150,120,90),5)
    #          , ((107,142,35),7), ((70,130,180),6), ((220,20,60),2), ((255,0,0),2), ((0,0,142),1), ((0,0,70),1)
    #          , ((0,60,100),1), ((0,0,90),1), ((0,0,110),1), ((0,0,230),3), ((119,11,32),3)]
    for k,v in trainId2label.items():
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def cityscape_maskmap_34Id():
    id2label        = { label.color:  label.id for label in labels           }
    rgb_to_maskidx = defaultdict(int)
    # ## id 1: car, license plate, truck, bus, caravan, trailer, 
    # ## id 2: person, rider
    # ## id 3: motorcycle, bicycle
    # ## id 4: road, sidewalk, parking, rail track, 
    # ## id 5: building, wall, fence, guard rail, bridge, tunnel
    # ## id 6: sky
    # ## id 7: vegetation
    # ## id 0: others
    # maps = [((128,64,128),4), ((244,35,232),4), ((250,170,160),4), ((230,150,140),4), ((70,70,70),5)
    #          , ((102,102,156),5), ((190,153,153),5), ((180,165,180),5), ((150,100,100),5), ((150,120,90),5)
    #          , ((107,142,35),7), ((70,130,180),6), ((220,20,60),2), ((255,0,0),2), ((0,0,142),1), ((0,0,70),1)
    #          , ((0,60,100),1), ((0,0,90),1), ((0,0,110),1), ((0,0,230),3), ((119,11,32),3)]
    for k,v in id2label.items():
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def cityscape_maskmap_8catId():
    color2catId      = { label.color: label.categoryId for label in labels }
    rgb_to_maskidx = defaultdict(int)
    # ## id 1: car, license plate, truck, bus, caravan, trailer, 
    # ## id 2: person, rider
    # ## id 3: motorcycle, bicycle
    # ## id 4: road, sidewalk, parking, rail track, 
    # ## id 5: building, wall, fence, guard rail, bridge, tunnel
    # ## id 6: sky
    # ## id 7: vegetation
    # ## id 0: others
    # maps = [((128,64,128),4), ((244,35,232),4), ((250,170,160),4), ((230,150,140),4), ((70,70,70),5)
    #          , ((102,102,156),5), ((190,153,153),5), ((180,165,180),5), ((150,100,100),5), ((150,120,90),5)
    #          , ((107,142,35),7), ((70,130,180),6), ((220,20,60),2), ((255,0,0),2), ((0,0,142),1), ((0,0,70),1)
    #          , ((0,60,100),1), ((0,0,90),1), ((0,0,110),1), ((0,0,230),3), ((119,11,32),3)]
    for k,v in color2catId.items():
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
def BDD_maskmap_8catId():
    color2catId      = { label.color: label.categoryId for label in labels }
    rgb_to_maskidx = defaultdict(int)
    # ## id 1: car, license plate, truck, bus, caravan, trailer, 
    # ## id 2: person, rider
    # ## id 3: motorcycle, bicycle
    # ## id 4: road, sidewalk, parking, rail track, 
    # ## id 5: building, wall, fence, guard rail, bridge, tunnel
    # ## id 6: sky
    # ## id 7: vegetation
    # ## id 0: others
    maps = [((0,  0,  0),0), ((111, 74,  0),0), ((0,  0,  0),0), ((81,  0, 81),0), ((0,  0,  0),0)
             , ((250, 170, 160),1), ((230, 150, 140),1), ((128, 64,128),1), ((244, 35,232),1), ((150, 100, 100),2)
             , ((70, 70, 70),2), ((190, 153, 153),2), ((180, 100, 180),2), ((180, 165, 180),2), ((150, 120, 90),2), ((102, 102, 156),2)
             , ((250, 170, 100),3), ((220, 220, 250),3), ((255, 165, 0),3), ((220, 20, 60),3), ((153, 153, 153),3)
             , ((153, 153, 153),3), ((220, 220,100),3), ((255, 70, 0),3), ((220, 220, 220),3), ((250, 170, 30),3)
             , ((220, 220, 0),3), ((250, 170, 250),3), ((152, 251, 152),4), ((107, 142, 35),4), ((70, 130, 180),5)
             , ((220, 20, 60),6), ((255, 0, 0),6), ((119, 11, 32),7), ((0, 60,100),7), ((0, 0,142),7)
             , ((0, 0, 90),7), ((0, 0, 230),7), ((0, 0, 110),7), ((0, 80, 100),7), ((0, 0, 70),7)]
    for k,v in maps:
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

# http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds
def TODO_virtual_kitti_maskmap():
    rgb_to_maskidx = defaultdict(int)
    # Category(:id) r g b
    # Terrain 210 0 200       4
    # Sky 90 200 255          5
    # Tree 0 199 0            4
    # Vegetation 90 240 0     4
    # Building 140 140 140    2
    # Road 100 60 100         1
    # GuardRail 255 100 255   2
    # TrafficSign 255 255 0   3
    # TrafficLight 200 200 0  3
    # Pole 255 130 0          3
    # Misc 80 80 80           -1-->0
    # Truck 160 60 60         7
    # Car:0 200 200 200       0-->7
    maps = [((128,64,128),4), ((244,35,232),4), ((250,170,160),4), ((230,150,140),4), ((70,70,70),5)
             , ((102,102,156),5), ((190,153,153),5), ((180,165,180),5), ((150,100,100),5), ((150,120,90),5)
             , ((107,142,35),7), ((70,130,180),6), ((220,20,60),2), ((255,0,0),2), ((0,0,142),1), ((0,0,70),1)
             , ((0,60,100),1), ((0,0,90),1), ((0,0,110),1), ((0,0,230),3), ((119,11,32),3)]
    for k,v in maps:
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

def TODO_synthia_seq_maskmap():
    color2catId      = { label.color: label.categoryId for label in labels }
    rgb_to_maskidx = defaultdict(int)
    # Misc        0
    # Sky         5
    # Building    2
    # Road        1
    # sidewalk    1
    # fence       2
    # Vegetation  4
    # Pole        3
    # car         7
    # sign        3
    # pedestrian  6
    # cyclist     6
    # lanemarking 1
    for k,v in color2catId.items():
        rgb_to_maskidx[k] = v
    return rgb_to_maskidx 

# def A_maskmap():
#     return cityscape()

# def B_maskmap():
#     return cityscape()

def preprocess_master(src, preprocess_func, file_regex="*.png"):
    dst = src.replace("_seg", DST_DIR_POSTFIX)
    if not os.path.exists(dst):
        os.makedirs(dst)
    segs = set(glob(os.path.join(src, file_regex)))
    pool = ProcessPool(8)
    pool.map(preprocess_func, segs)
 
def cityscape_preprocess(image_seg):
    if not os.path.exists(image_seg.replace("_seg", DST_DIR_POSTFIX)):
        base_name = os.path.basename(image_seg)
        print "processing", base_name
        img = imread(image_seg)
        M, N = img.shape[:2]
        seg_class = np.zeros((M, N)).astype(np.int)
        for x in range(M):
            for y in range(N):
                seg_class[x,y] = maskmap[tuple(img[x,y,:3])]
        toimage(seg_class, cmin=0, cmax=255).save(image_seg.replace("_seg", DST_DIR_POSTFIX)) 
    else:
        print "skip"
 
def virtual_kitti_preprocess(image_seg):
    base_name = os.path.basename(image_seg)
    print "processing", base_name
    img = imread(image_seg)
    M, N = img.shape[:2]
    seg_class = np.ones((M, N)).astype(np.int) * 1
    for x in range(M):
        for y in range(N):
            seg_class[x,y] = maskmap[tuple(img[x,y,:3])]
    toimage(seg_class, cmin=0, cmax=255).save(image_seg.replace("_seg", DST_DIR_POSTFIX)) 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gta", type=str, help="dataset name")
    parser.add_argument('--use_8catId', type=str2bool, default=False, help='use 8catId or 20trainId')
    parser.add_argument('--file_regex', type=str, default="*.png", help='use 8catId or 20trainId')
    args = vars(parser.parse_args())

    if args["dataset"].lower() in ['gta', 'gta25k', 'viper', 'synsf', 'cityscapes', 'cityscape']:
        if args['use_8catId']:
            DST_DIR_POSTFIX = "_seg_class_8catId"
            maskmap = cityscape_maskmap_8catId()
        else:
            maskmap = cityscape_maskmap_20trainId()
            DST_DIR_POSTFIX = "_seg_class_20trainId"
        preprocess_func = cityscape_preprocess
    elif args["dataset"].lower() in ['gta25k_bdd']:
        if args['use_8catId']:
            DST_DIR_POSTFIX = "_seg_class_8catId"
            maskmap = BDD_maskmap_8catId()
        else:
            maskmap = cityscape_maskmap_20trainId()
            DST_DIR_POSTFIX = "_seg_class_20trainId"
        preprocess_func = cityscape_preprocess


    elif args["dataset"].lower() in ['kitti_PhilippeXu']:
        maskmap = kitti_PhilippeXu_maskmap()
        preprocess_func = kitti_PhilippeXu_preprocess
    elif args["dataset"].lower() in ['virtual_kitti']:
        maskmap = virtual_kitti_maskmap()
        preprocess_func = virtual_kitti_preprocess
    else:
        raise Exception('args["dataset"] is not a suitable dataset name')

    # pdb.set_trace() 
    preprocess_master("./{}/trainB_seg".format(args["dataset"]), preprocess_func, args["file_regex"])
    preprocess_master("./{}/testB_seg".format(args["dataset"]), preprocess_func, args["file_regex"])
    # pdb.set_trace()
    # preprocess_master("./{}/testA_seg".format(args["dataset"]), preprocess_func, args["file_regex"])
    # preprocess_master("./{}/testB_seg".format(args["dataset"]), preprocess_func, args["file_regex"])
