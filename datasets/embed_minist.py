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

data_dir = '/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data'
dataset = 'mnist'
A_dir = os.path.join(data_dir, dataset, 'testA')
B_dir = os.path.join(data_dir, dataset, 'testB')
sample_num = 1000
_IMG_PATTERN_list = ['*.png','*.jpg']
np.random.seed(0)

pathlistA, pathlistB = [], []
for _IMG_PATTERN in _IMG_PATTERN_list:
    pathlistA += glob.glob(os.path.join(A_dir, _IMG_PATTERN))
    pathlistB += glob.glob(os.path.join(B_dir, _IMG_PATTERN))
np.random.shuffle(pathlistA)
np.random.shuffle(pathlistB)

X_list_A, X_list_B = [], []
y_list_A, y_list_B = [], []
H, W, C = scipy.misc.imread(pathlistA[0]).shape
for i in range(sample_num):
    X_list_A.append(scipy.misc.imread(pathlistA[i]).reshape(-1))
    y_list_A.append(int(pathlistA[i].split('_')[-1].split('.')[0]))
    X_list_B.append(scipy.misc.imread(pathlistB[i]).reshape(-1))
    y_list_B.append(int(pathlistB[i].split('_')[-3]))

X = np.stack(X_list_B, axis=0)
y = np.stack(y_list_B, axis=0)
X_img = X.reshape(X.shape[0], H, W, C)
X = X.astype(np.float)
X = decomposition.TruncatedSVD(n_components=28*28).fit_transform(X)


# H, W, C = 8, 8, 1
# for i in range(sample_num):
#     X_list_A.append(scipy.misc.imresize(scipy.misc.imread(pathlistA[i]), [H, W])[:,:,0].reshape(-1))
#     y_list_A.append(int(pathlistA[i].split('_')[-1].split('.')[0]))
#     X_list_B.append(scipy.misc.imresize(scipy.misc.imread(pathlistB[i]), [H, W])[:,:,0].reshape(-1))
#     y_list_B.append(int(pathlistB[i].split('_')[-3]))
# X_A = np.stack(X_list_A, axis=0)
# y_A = np.stack(y_list_A, axis=0)

# X = X_A
# y = y_A
# X_img = X.reshape(X.shape[0], H, W, C)
# X = X.astype(np.float)

# digits = datasets.load_digits(n_class=6)
# X = digits.data
# y = digits.target
# X_img = digits.images
# pdb.set_trace()

n_samples, n_features = X.shape
n_neighbors = 30


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None, plotImg=True):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
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
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne, y,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.savefig(os.path.join(data_dir, dataset, 't-SNE.png'))
plt.savefig(os.path.join(data_dir, dataset, 't-SNE.pdf'))
plt.show()