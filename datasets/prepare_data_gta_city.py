import sys
import os
import argparse
from glob import glob
import random
import shutil, pdb
random.seed(0)

def prepare_cs(img_dir, seg_dir, img_target_dir, seg_target_dir, tr_num, ts_num, replace_names=None):
    imgs = set(glob(os.path.join(img_dir, "*.png")))
    segs = set(glob(os.path.join(seg_dir, "*.png")))
    pairs = []
    for img_path in list(imgs):
        seg_path = os.path.join(seg_dir, (img_path.split("/")[-1].replace(replace_names[0], replace_names[1]) if replace_names else img_path.split("/")[-1]))
        # pdb.set_trace()
        if seg_path in segs:
            pairs.append((img_path, seg_path))
    print "candidates:", len(pairs)
    if len(pairs) < tr_num + ts_num:
        print "Cityscape candidates not enough! only %d"%len(pairs)
        tr_num = len(pairs)
        # return
    if not os.path.exists(img_target_dir):
        os.makedirs(img_target_dir)
        os.makedirs(img_target_dir.replace("train", "test"))
    if not os.path.exists(seg_target_dir):
        os.makedirs(seg_target_dir)
        os.makedirs(seg_target_dir.replace("train", "test"))
    random.shuffle(pairs)
    for i in range(tr_num):
        dst_path = img_target_dir + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][0], dst_path)
        dst_path = seg_target_dir + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][1], dst_path)
    for i in range(tr_num, tr_num+ts_num):
        dst_path = img_target_dir.replace("train", "test") + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][0], dst_path)
        dst_path = seg_target_dir.replace("train", "test") + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][1], dst_path)

def prepare_gta(img_dir, seg_dir, img_target_dir, seg_target_dir, tr_num, ts_num, replace_names=None):
    imgs = set(glob(os.path.join(img_dir, "*.png")))
    segs = set(glob(os.path.join(seg_dir, "*.png")))
    pairs = []
    for img_path in list(imgs):
        seg_path = os.path.join(seg_dir, (img_path.split("/")[-1].replace(replace_names[0], replace_names[1]) if replace_names else img_path.split("/")[-1]))
        # pdb.set_trace()
        if seg_path in segs:
            pairs.append((img_path, seg_path))
    print "candidates:", len(pairs)
    if len(pairs) < tr_num + ts_num:
        print "gta candidates not enough!"
        return
    if not os.path.exists(img_target_dir):
        os.makedirs(img_target_dir)
        os.makedirs(img_target_dir.replace("train", "test"))
    if not os.path.exists(seg_target_dir):
        os.makedirs(seg_target_dir)
        os.makedirs(seg_target_dir.replace("train", "test"))
    random.shuffle(pairs)
    for i in range(tr_num):
        dst_path = img_target_dir + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][0], dst_path)
        dst_path = seg_target_dir + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][1], dst_path)
    for i in range(tr_num, tr_num+ts_num):
        dst_path = img_target_dir.replace("train", "test") + (pairs[i][0].split("/")[-1].replace(replace_names[0], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][0], dst_path)
        dst_path = seg_target_dir.replace("train", "test") + (pairs[i][1].split("/")[-1].replace(replace_names[1], "") if replace_names else "")
        if not os.path.isfile(dst_path):
            shutil.copy2(pairs[i][1], dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--A_imagepath", "-Ai", type=str, default="/home/lpl/data/playing/images/", help="dataset A's image path")
    parser.add_argument("--A_segpath", "-As", type=str, default="/home/lpl/data/playing/labels/", help="dataset A's segmentation path")
    # cp `find train/ -name "*.png"` all_train/
    parser.add_argument("--B_imagepath", "-Bi", type=str, default="/home/lpl/data/cityscape/leftImg8bit/all_train/", help="dataset B's image path")
    parser.add_argument("--B_segpath", "-Bs", type=str, default="/home/lpl/data/cityscape/gtFine/all_train/", help="dataset B's segmentation path")
    parser.add_argument("--train_size", "-tr", type=int, default=24466, help="number of training examples for each dataset")
    parser.add_argument("--test_size", "-te", type=int, default=500, help="number of test examples for each dataset")
    args = vars(parser.parse_args())

    prepare_gta(args["A_imagepath"], args["A_segpath"], "./gta25k/trainA/", "./gta25k/trainA_seg/", args["train_size"], args["test_size"])
    prepare_cs(args["B_imagepath"], args["B_segpath"], "./gta25k/trainB/", "./gta25k/trainB_seg/", args["train_size"], args["test_size"], replace_names=("_leftImg8bit", "_gtFine_color"))

