import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cs_seg_eval
import os, glob, pdb
import numpy as np
import scipy.misc
from labels_utils import *

def trainId2Id(mask):
    trainId2Id = { label.trainId: label.id for label in labels }
    trainId2Id[19] = 0 ## ignored during eval
    img_h, img_w = mask.shape
    for h in range(img_h):
        for w in range(img_w):
            mask[h,w] = trainId2Id[mask[h,w]]
    return mask

groundTruthImgList = []
for i in range(4):
    src1 = 'logs/MODEL2001tmp_gta25k20_segment_LinkNet_csEval_256x512_bs4_flip_crop_bright_lr2e-4/sample/groundTruthImg%02d.png'%i
    src2 = 'logs/MODEL2001tmp_gta25k20_segment_LinkNet_csEval_256x512_bs4_flip_crop_bright_lr2e-4/sample/predictionImg%02d.png'%i
    img1 = scipy.misc.imread(src1)
    img2 = scipy.misc.imread(src2)
    pdb.set_trace()

    # dst = trainId2Id(scipy.misc.imread(src))
    # path = 'tmp/1.png'
    # scipy.misc.imsave(path, dst)
    groundTruthImgList.append(src)

predictionImgList = []
for i in range(4):
    src = 'logs/MODEL2001tmp_gta25k20_segment_LinkNet_csEval_256x512_bs4_flip_crop_bright_lr2e-4/sample/predictionImg%02d.png'%i
    # dst = trainId2Id(scipy.misc.imread(src))
    # path = 'tmp/2.png'
    # scipy.misc.imsave(path, dst)
    predictionImgList.append(src)


# for i in range(1):
#     # pred
#     path = '{}/predictionImg{:02d}.jpg'.format(sample_dir,i)
#     predictionImgList.append(path)
#     # pdb.set_trace()
#     # imsave(trainId2Id(np.expand_dims(pred_mask_B_ts[i,:,:],0)), [1,1], path)
#     pred = trainId2Id(pred_mask_B_ts[i,:,:])
#     scipy.misc.imsave(path, pred)
#     # gt
#     path = '{}/groundTruthImg{:02d}.jpg'.format(sample_dir,i)
#     groundTruthImgList.append(path)
#     # imsave(trainId2Id(np.expand_dims(mask_B_ts_ori[i,:,:],0)), [1,1], path)
#     # pdb.set_trace()
#     gt = trainId2Id(mask_B_ts_ori[i,:,:])
#     scipy.misc.imsave(path, gt)
# CSUPPORT = False
cs_seg_eval.evaluateImgLists(predictionImgList, groundTruthImgList, cs_seg_eval.args)
# # cs_seg_eval.evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instanceStats, perImageStats, args)
