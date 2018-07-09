# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011
## ref to http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py

# print(__doc__)
from time import time

import os, pdb, glob, scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from skimage.measure import compare_ssim as ssim

# Set gpu and import tf
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
# import tflib
# import tflib.inception_score


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, X_img=None, title=None, plotImg=True):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    # pdb.set_trace()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]

            if plotImg:
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(X_img[i].reshape(H, W, C), cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


#----------------------------------------------------------------------
# # Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


# #----------------------------------------------------------------------
# # Random 2D projection using a random unitary matrix
# print("Computing random projection")
# rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
# X_projected = rp.fit_transform(X)
# plot_embedding(X_projected, y, "Random Projection of the digits")


# #----------------------------------------------------------------------
# # Projection on to the first 2 principal components

# print("Computing PCA projection")
# t0 = time()
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
# plot_embedding(X_pca, y,
#                "Principal Components projection of the digits (time %.2fs)" %
#                (time() - t0))

# #----------------------------------------------------------------------
# # Projection on to the first 2 linear discriminant components

# print("Computing Linear Discriminant Analysis projection")
# X2 = X.copy()
# X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
# t0 = time()
# X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
# plot_embedding(X_lda, y,
#                "Linear Discriminant projection of the digits (time %.2fs)" %
#                (time() - t0))


# #----------------------------------------------------------------------
# # Isomap projection of the digits dataset
# print("Computing Isomap embedding")
# t0 = time()
# X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
# print("Done.")
# plot_embedding(X_iso, y,
#                "Isomap projection of the digits (time %.2fs)" %
#                (time() - t0))


# #----------------------------------------------------------------------
# # Locally linear embedding of the digits dataset
# print("Computing LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='standard')
# t0 = time()
# X_lle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_lle, y,
#                "Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# #----------------------------------------------------------------------
# # Modified Locally linear embedding of the digits dataset
# print("Computing modified LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='modified')
# t0 = time()
# X_mlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_mlle, y,
#                "Modified Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# #----------------------------------------------------------------------
# # HLLE embedding of the digits dataset
# print("Computing Hessian LLE embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='hessian')
# t0 = time()
# X_hlle = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_hlle, y,
#                "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
#                (time() - t0))


# #----------------------------------------------------------------------
# # LTSA embedding of the digits dataset
# print("Computing LTSA embedding")
# clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
#                                       method='ltsa')
# t0 = time()
# X_ltsa = clf.fit_transform(X)
# print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
# plot_embedding(X_ltsa, y,
#                "Local Tangent Space Alignment of the digits (time %.2fs)" %
#                (time() - t0))

# #----------------------------------------------------------------------
# # MDS  embedding of the digits dataset
# print("Computing MDS embedding")
# clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
# t0 = time()
# X_mds = clf.fit_transform(X)
# print("Done. Stress: %f" % clf.stress_)
# plot_embedding(X_mds, y,
#                "MDS embedding of the digits (time %.2fs)" %
#                (time() - t0))

# #----------------------------------------------------------------------
# # Random Trees embedding of the digits dataset
# print("Computing Totally Random Trees embedding")
# hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                        max_depth=5)
# t0 = time()
# X_transformed = hasher.fit_transform(X)
# pca = decomposition.TruncatedSVD(n_components=2)
# X_reduced = pca.fit_transform(X_transformed)

# plot_embedding(X_reduced, y,
#                "Random forest embedding of the digits (time %.2fs)" %
#                (time() - t0))

# #----------------------------------------------------------------------
# # Spectral embedding of the digits dataset
# print("Computing Spectral embedding")
# embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                       eigen_solver="arpack")
# t0 = time()
# X_se = embedder.fit_transform(X)

# plot_embedding(X_se, y,
#                "Spectral embedding of the digits (time %.2fs)" %
#                (time() - t0))

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
def t_SNE(data_dir, dataset, X_A, X_B, X_ab, X_ba, y_A, y_B, y_ab, y_ba):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    t0 = time()
    X = np.concatenate((X_A,X_ba), axis=0)
    y = np.concatenate((y_A,y_ba), axis=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, y,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0), plotImg=False)
    plt.savefig(os.path.join(data_dir, dataset, 't-SNE_(X_A,X_ba).png'))
    plt.savefig(os.path.join(data_dir, dataset, 't-SNE_(X_A,X_ba).pdf'))

    t0 = time()
    X = np.concatenate((X_B,X_ab), axis=0)
    y = np.concatenate((y_B,y_ab), axis=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, y,
                   "t-SNE embedding of the digits (time %.2fs)" %
                   (time() - t0), plotImg=False)
    plt.savefig(os.path.join(data_dir, dataset, 't-SNE_(X_B,X_ab).png'))
    plt.savefig(os.path.join(data_dir, dataset, 't-SNE_(X_B,X_ab).pdf'))
    # plt.show()


def pix_acc(data_dir, dataset, X_img_A, X_img_B, X_img_ab, X_img_ba, threshold_list=[1,2,4,8,16]):
    ba_acc_list, ab_acc_list = [], []
    with open(os.path.join(data_dir, dataset, 'pixel_acc.txt'), "w") as f:
        for threshold in threshold_list:
            # pdb.set_trace()
            ba_error = np.sum(abs(X_img_ba - X_img_A), axis=-1, keepdims=False)
            ba_error_idxs = np.where(ba_error > threshold) 
            ab_error = np.sum(abs(X_img_ab - X_img_B), axis=-1, keepdims=False)
            ab_error_idxs = np.where(ab_error > threshold)
            N, H, W, C = X_img_A.shape
            ba_acc = 1 - float(len(ba_error_idxs[0]))/float(N*H*W)
            ab_acc = 1 - float(len(ab_error_idxs[0]))/float(N*H*W)
            ba_acc_list.append(ba_acc)
            ab_acc_list.append(ab_acc)
            # print("ba_Accuracy: %f\n" % ba_acc)
            # print("ab_Accuracy: %f\n" % ab_acc)
            f.write("############################\n")
            f.write("Threshold: %d\n" % threshold)
            f.write("ba_Accuracy: %f\n" % ba_acc)
            f.write("ab_Accuracy: %f\n" % ab_acc)

    plt.figure()
    plt.plot(threshold_list, ba_acc_list, 'r')
    plt.plot(threshold_list, ab_acc_list, 'b')
    plt.xlim([0, 100])
    plt.ylim([0.0, 1.0])
    plt.xlabel('threshold')
    plt.ylabel('pixel acc')
    # plt.title()
    plt.savefig(os.path.join(data_dir, dataset, 'pixel_acc.png'))
    plt.savefig(os.path.join(data_dir, dataset, 'pixel_acc.pdf'))

def inception_score(data_dir, dataset, X_img_list_A, X_img_list_B, X_img_list_ab, X_img_list_ba):
    ##################### Inception score ##################
    IS_A_mean, IS_A_std = tflib.inception_score.get_inception_score(X_img_list_A)
    IS_B_mean, IS_B_std = tflib.inception_score.get_inception_score(X_img_list_B)
    IS_ab_mean, IS_ab_std = tflib.inception_score.get_inception_score(X_img_list_ab)
    IS_ba_mean, IS_ba_std = tflib.inception_score.get_inception_score(X_img_list_ba)
    with open(os.path.join(data_dir, dataset, 'inception_score.txt'), "w") as f:
        f.write("IS_A_mean: %f, IS_A_std:%f\n" % (IS_A_mean, IS_A_std))
        f.write("IS_B_mean: %f, IS_B_std:%f\n" % (IS_B_mean, IS_B_std))
        f.write("IS_ab_mean: %f, IS_ab_std:%f\n" % (IS_ab_mean, IS_ab_std))
        f.write("IS_ba_mean: %f, IS_ba_std:%f\n" % (IS_ba_mean, IS_ba_std))

def ssim_score(data_dir, dataset, X_img_A, X_img_B, X_img_ab, X_img_ba, multichannel=True):
    N, H, W, C = X_img_A.shape
    ba_ssim_list, ab_ssim_list = [], []
    for i in range(N):
        mssim = ssim(X_img_ba[i,:,:,:], X_img_A[i,:,:,:], multichannel=multichannel) # multichannel result should be mean of the individual channel results
        ba_ssim_list.append(mssim)
        mssim = ssim(X_img_ab[i,:,:,:], X_img_B[i,:,:,:], multichannel=multichannel) # multichannel result should be mean of the individual channel results
        ab_ssim_list.append(mssim)
    with open(os.path.join(data_dir, dataset, 'ssim_score.txt'), "w") as f:
        f.write("############################\n")
        f.write("multichannel: %r\n" % multichannel)
        f.write("ba_ssim: %f\n" % np.mean(ba_ssim_list))
        f.write("ab_ssim: %f\n" % np.mean(ab_ssim_list))


def error_map(data_dir, dataset, X_img_A, X_img_B, X_img_ab, X_img_ba, use_abs=True):
    N, H, W, C = X_img_A.shape
    if use_abs:
        ba_error = np.tile(np.mean(abs(X_img_ba - X_img_A), axis=-1, keepdims=True), [1,1,1,3])
        ab_error = np.tile(np.mean(abs(X_img_ab - X_img_B), axis=-1, keepdims=True), [1,1,1,3])
        ba_dir = os.path.join(data_dir, dataset, 'ba_error_use_abs')
        ab_dir = os.path.join(data_dir, dataset, 'ab_error_use_abs')
    else:
        ba_error = (X_img_ba - X_img_A)/2.0 + 127.5
        ab_error = (X_img_ab - X_img_B)/2.0 + 127.5
        ba_dir = os.path.join(data_dir, dataset, 'ba_error')
        ab_dir = os.path.join(data_dir, dataset, 'ab_error')
    if not os.path.exists(ba_dir):
        os.makedirs(ba_dir)
    if not os.path.exists(ab_dir):
        os.makedirs(ab_dir)
    for i in range(N):
        scipy.misc.imsave(os.path.join(ba_dir, '%06d.png'%i), ba_error[i,:,:,:])
        scipy.misc.imsave(os.path.join(ab_dir, '%06d.png'%i), ab_error[i,:,:,:])

def main(data_dir, dataset, GT_dir):
    # Use_Transfered_GT = False
    # if Use_Transfered_GT:
    # else:
    testA2B_dir = os.path.join(GT_dir, 'testA2B')
    testB2A_dir = os.path.join(GT_dir, 'testB2A')
    A_dir = os.path.join(data_dir, dataset, 'A')
    B_dir = os.path.join(data_dir, dataset, 'B')
    ab_dir = os.path.join(data_dir, dataset, 'ab')
    ba_dir = os.path.join(data_dir, dataset, 'ba')
    sample_num = 1000
    _IMG_PATTERN_list = ['*.png','*.jpg']
    np.random.seed(0)

    pathlistA, pathlistB, pathlistA2B, pathlistB2A, pathlist_ab, pathlist_ba = [], [], [], [] , [], []
    for _IMG_PATTERN in _IMG_PATTERN_list:
        pathlistA += sorted(glob.glob(os.path.join(A_dir, _IMG_PATTERN)))
        pathlistB += sorted(glob.glob(os.path.join(B_dir, _IMG_PATTERN)))
        pathlistA2B += sorted(glob.glob(os.path.join(testA2B_dir, _IMG_PATTERN)))
        pathlistB2A += sorted(glob.glob(os.path.join(testB2A_dir, _IMG_PATTERN)))
        pathlist_ab += sorted(glob.glob(os.path.join(ab_dir, _IMG_PATTERN)))
        pathlist_ba += sorted(glob.glob(os.path.join(ba_dir, _IMG_PATTERN)))
    # np.random.shuffle(pathlistA)
    # np.random.shuffle(pathlistB)
    # np.random.shuffle(pathlist_ab)
    # np.random.shuffle(pathlist_ba)

    X_list_A, X_list_B, X_list_A2B, X_list_B2A, X_list_ab, X_list_ba = [], [], [], [], [], []
    y_list_A, y_list_B, y_list_A2B, y_list_B2A, y_list_ab, y_list_ba = [1]*sample_num, [1]*sample_num, [1]*sample_num, [1]*sample_num, [2]*sample_num, [2]*sample_num
    H, W, C = scipy.misc.imread(pathlistA[0]).shape
    for i in range(sample_num):
        X_list_A.append(scipy.misc.imread(pathlistA[i]).reshape(-1))
        X_list_B.append(scipy.misc.imread(pathlistB[i]).reshape(-1))
        X_list_A2B.append(scipy.misc.imread(pathlistA2B[i]).reshape(-1))
        X_list_B2A.append(scipy.misc.imread(pathlistB2A[i]).reshape(-1))
        X_list_ab.append(scipy.misc.imread(pathlist_ab[i]).reshape(-1))
        X_list_ba.append(scipy.misc.imread(pathlist_ba[i]).reshape(-1))

    X_A = np.stack(X_list_A, axis=0)
    y_A = np.stack(y_list_A, axis=0)
    X_img_A = X_A.reshape((X_A.shape[0], H, W, C))
    X_A = X_A.astype(np.float)

    X_B = np.stack(X_list_B, axis=0)
    y_B = np.stack(y_list_B, axis=0)
    X_img_B = X_B.reshape((X_B.shape[0], H, W, C))
    X_B = X_B.astype(np.float)

    X_A2B = np.stack(X_list_A2B, axis=0)
    y_A2B = np.stack(y_list_A2B, axis=0)
    X_img_A2B = X_A2B.reshape((X_A2B.shape[0], H, W, C))
    X_A2B = X_A2B.astype(np.float)

    X_B2A = np.stack(X_list_B2A, axis=0)
    y_B2A = np.stack(y_list_B2A, axis=0)
    X_img_B2A = X_B2A.reshape((X_B2A.shape[0], H, W, C))
    X_B2A = X_B2A.astype(np.float)

    X_ab = np.stack(X_list_ab, axis=0)
    y_ab = np.stack(y_list_ab, axis=0)
    X_img_ab = X_ab.reshape((X_ab.shape[0], H, W, C))
    X_ab = X_ab.astype(np.float)

    X_ba = np.stack(X_list_ba, axis=0)
    y_ba = np.stack(y_list_ba, axis=0)
    X_img_ba = X_ba.reshape((X_ba.shape[0], H, W, C))
    X_ba = X_ba.astype(np.float)

    # n_samples, n_features = X_A.shape
    # n_neighbors = 30
    # inception_score(data_dir, dataset, [X_img_A[i] for i in range(sample_num)], [X_img_B[i] for i in range(sample_num)], [X_img_ab[i] for i in range(sample_num)], [X_img_ba[i] for i in range(sample_num)])
    pix_acc(data_dir, dataset, X_img_B2A, X_img_A2B, X_img_ab, X_img_ba, threshold_list=range(100))
    ssim_score(data_dir, dataset, X_img_A, X_img_B, X_img_ab, X_img_ba, multichannel=True)
    error_map(data_dir, dataset, X_img_B2A, X_img_A2B, X_img_ab, X_img_ba, use_abs=True)
    # t_SNE(data_dir, dataset, X_A, X_B, X_ab, X_ba, y_A, y_B, y_ab, y_ba)

def t_SNE_all(data_dir, dataset_list, sample_num=1000, direction='A2B', use_pca=True):
    _IMG_PATTERN_list = ['*.png','*.jpg']
    X_list, Res_list = [], []
    ## X list
    # np.random.seed(0)
    if 'A2B'==direction:
        X_dir = os.path.join(data_dir, dataset_list[0], 'B')
    else:
        X_dir = os.path.join(data_dir, dataset_list[0], 'A')
    pathlistX = []
    for _IMG_PATTERN in _IMG_PATTERN_list:
        pathlistX += sorted(glob.glob(os.path.join(X_dir, _IMG_PATTERN)))
    for i in range(sample_num):
        X_list.append(scipy.misc.imread(pathlistX[i]).reshape(-1))
    X = np.stack(X_list, axis=0)

    ## Res list
    for dataset in dataset_list:
        if 'A2B'==direction:
            Res_dir = os.path.join(data_dir, dataset, 'ab')
        else:
            Res_dir = os.path.join(data_dir, dataset, 'ba')
        pathlistRes = []
        for _IMG_PATTERN in _IMG_PATTERN_list:
            pathlistRes += sorted(glob.glob(os.path.join(Res_dir, _IMG_PATTERN)))
        print dataset
        for i in range(sample_num):
            Res_list.append(scipy.misc.imread(pathlistRes[i]).reshape(-1))
    Res = np.stack(Res_list, axis=0)


    ## t-sne
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_in = np.concatenate((X, Res), axis=0)
    if use_pca:
        X_in = decomposition.TruncatedSVD(n_components=50).fit_transform(X_in)
    X_tsne = tsne.fit_transform(X_in)
    # pdb.set_trace()
    X_tsne_list = np.split(X_tsne, len(dataset_list)+1, axis=0)

    ## plot
    for i in range(len(dataset_list)):
        plot_embedding(np.concatenate((X_tsne_list[0], X_tsne_list[i+1]), axis=0), np.stack(([1]*sample_num+[2]*sample_num), axis=0), plotImg=False)
        # pdb.set_trace()
        if 'A2B'==direction:
            if use_pca:
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ab)_pca50.png'))
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ab)_pca50.pdf'))
            else:
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ab).png'))
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ab).pdf'))
        else:
            if use_pca:
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ba)_pca50.png'))
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ba)_pca50.pdf'))
            else:
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ba).png'))
                plt.savefig(os.path.join(data_dir, dataset_list[i], 't-SNE_all_(X_A,X_ba).pdf'))


if __name__ == '__main__':
    ## SG-UNIT
    data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/SG-GAN/logs_rightGammaBeta/'
    # # # dataset = 'MODEL1099_mnist_BW_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_1G1D_epoch6/test_mnist_BW_test_28x28_88k'
    # # # dataset = 'MODEL1099_mnist_BW_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist'
    # dataset = 'MODEL1099_mnist_BW_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_56k'
    # # dataset = 'MODEL1097_mnist_BW_changeRes_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style'
    # # dataset = 'MODEL1093_mnist_BW_changeRes_feaMask_1Conv_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style'
    # # dataset = 'MODEL1500_mnist_BW_CycleGAN_changeRes_feaMask_1Conv_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style'
    # # dataset = 'MODEL1000_mnist_BW_unit_noNorm_InsNormDis_1Conv_100L1_lsgan_256x512_bs8_lr1e-5_5G1D/test_mnist_BW_test_28x28_56k'
    # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_BW'

    # dataset = 'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_56k'
    # dataset = 'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch6/test_mnist_multi_jitterColor_BW_test_112x112_176k'
    # dataset = 'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e3L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch6/test_mnist_multi_jitterColor_BW_test_112x112_120k'
    dataset = 'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_56k'
    # dataset = 'MODEL1000_mnist_multi_jitterColor_BW_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_lr1e-5_5G1D/test_mnist_multi_jitterColor_BW_test_112x112_56k'
    GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_multi_jitterColor_BW'

    ## MUNIT
    # data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/MUNIT/results'
    # dataset = 'mnist_BW_vgg_5G1D_300k_1style'
    # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_BW'
    # # dataset = 'mnist_multi_jitterColor_BW_vgg_5G1D_300k_1style'
    # # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_multi_jitterColor_BW'


    # data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/SG-GAN/logs_rightGammaBeta/'
    # dataset = 'MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style'
    # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_multi_jitterColor_BW'
    # main(data_dir, dataset, GT_dir)

    # data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/SG-GAN/logs_rightGammaBeta/'
    # dataset = 'MODEL1097_mnist_multi_jitterColor_BW_changeRes_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style'
    # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_multi_jitterColor_BW'
    # main(data_dir, dataset, GT_dir)

    # data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/SG-GAN/logs_rightGammaBeta/'
    # dataset = 'MODEL1503_mnist_BW_changeRes_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style'
    # GT_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/data/mnist_BW'
    # main(data_dir, dataset, GT_dir)

    use_pca = False
    data_dir = '/BS/sun_project2/work/mlq_project/WassersteinGAN/'
    dataset_list = [
                    'SG-GAN/logs_rightGammaBeta/MODEL1099_mnist_BW_changeRes_feaMask_1Conv_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_56k',
                    'SG-GAN/logs_rightGammaBeta/MODEL1500_mnist_BW_CycleGAN_changeRes_feaMask_1Conv_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1502_mnist_BW_feaMask_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1097_mnist_BW_changeRes_1e3L1VggStyle5L_1e1L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1099_mnist_BW_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_BW_test_28x28_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1000_mnist_BW_unit_noNorm_InsNormDis_1Conv_100L1_lsgan_256x512_bs8_lr1e-5_5G1D/test_mnist_BW_test_28x28_56k',
                    'MUNIT/results/mnist_BW_vgg_5G1D_300k_1style',
                    ]
    t_SNE_all(data_dir, dataset_list, sample_num=1000, direction='A2B', use_pca=use_pca)
    t_SNE_all(data_dir, dataset_list, sample_num=1000, direction='B2A', use_pca=use_pca)
    dataset_list = [
                    'SG-GAN/logs_rightGammaBeta/MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_56k',
                    'SG-GAN/logs_rightGammaBeta/MODEL1500_mnist_multi_jitterColor_BW_CycleGAN_changeRes_feaMask_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1502_mnist_multi_jitterColor_BW_feaMask_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1097_mnist_multi_jitterColor_BW_changeRes_1e4L1VggStyle5L_1e2L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1099_mnist_multi_jitterColor_BW_changeRes_feaMask_0L1VggStyle5L_0L1ContentLoss5LLinearWeight_1INlayer_bs8_lr1e-5_5G1D_epoch1/test_mnist_multi_jitterColor_BW_test_112x112_1style',
                    'SG-GAN/logs_rightGammaBeta/MODEL1000_mnist_multi_jitterColor_BW_unit_noNorm_InsNormDis_100L1_lsgan_256x512_bs8_lr1e-5_5G1D/test_mnist_multi_jitterColor_BW_test_112x112_56k',
                    'MUNIT/results/mnist_multi_jitterColor_BW_vgg_5G1D_300k_1style',
                    ]
    t_SNE_all(data_dir, dataset_list, sample_num=1000, direction='A2B', use_pca=use_pca)
    t_SNE_all(data_dir, dataset_list, sample_num=1000, direction='B2A', use_pca=use_pca)