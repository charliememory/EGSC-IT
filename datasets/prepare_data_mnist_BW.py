import mnist, os, sys, pdb, tqdm
import scipy.misc
import numpy as np

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

color_Black = [0,0,0]
color_White = [255,255,255]
color_R = [255,0,0]
color_G = [0,255,0]
color_B = [0,0,255]
np.random.seed(0)

data_dir = int(sys.argv[1])
IsTrain = int(sys.argv[2])
if IsTrain:
    A_dir = os.path.join(data_dir, 'trainA')
    B_dir = os.path.join(data_dir, 'trainB')
    data_images = train_images
    data_labels = train_labels
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    if not os.path.exists(B_dir):
        os.makedirs(B_dir)
else:
    A_dir = os.path.join(data_dir, 'testA')
    B_dir = os.path.join(data_dir, 'testB')
    ab_dir = os.path.join(data_dir, 'testA2B')
    ba_dir = os.path.join(data_dir, 'testB2A')
    data_images = test_images
    data_labels = test_labels
    if not os.path.exists(A_dir):
        os.makedirs(A_dir)
    if not os.path.exists(B_dir):
        os.makedirs(B_dir)
    if not os.path.exists(ab_dir):
        os.makedirs(ab_dir)
    if not os.path.exists(ba_dir):
        os.makedirs(ba_dir)
idx_list_A = range(data_images.shape[0])
idx_list_B = range(data_images.shape[0])
np.random.shuffle(idx_list_A)
np.random.shuffle(idx_list_B)
rand_num_list_A = np.random.uniform(0.0,1.0, data_images.shape[0])
rand_num_list_B = np.random.uniform(0.0,1.0, data_images.shape[0])


## train_A
for i in tqdm.tqdm(range(data_images.shape[0])):
# for i in range(1):
    img = data_images[idx_list_A[i],:,:]
    label = data_labels[idx_list_A[i]]
    img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
    # rand_num = np.random.uniform(0.0,1.0)
    rand_num = rand_num_list_A[i]
    if rand_num < 0.5:
        # pdb.set_trace()
        r_idx, c_idx = np.where(img > 127.5)
        img_rgb[r_idx, c_idx, :] = color_Black
        r_idx, c_idx = np.where(img < 127.5)
        img_rgb[r_idx, c_idx, :] = color_White
        scipy.misc.toimage(img_rgb).save(os.path.join(A_dir, '%06d_%d_Black_Whilte.png'%(i,label)))
    else:
        r_idx, c_idx = np.where(img > 127.5)
        img_rgb[r_idx, c_idx, :] = color_White
        r_idx, c_idx = np.where(img < 127.5)
        img_rgb[r_idx, c_idx, :] = color_Black
        scipy.misc.toimage(img_rgb).save(os.path.join(A_dir, '%06d_%d_Whilte_Black.png'%(i,label)))

## train_B
rand_num_list = []
for i in tqdm.tqdm(range(data_images.shape[0])):
# for i in range(1):
    img = data_images[idx_list_B[i],:,:]
    label = data_labels[idx_list_B[i]]
    img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
    # rand_num = np.random.uniform(0,1)
    rand_num = rand_num_list_B[i]
    if IsTrain and label > 4:  ## In order to show the model can also transfer the style of unseen samples
        ## Foreground: R, Background:G
        r_idx, c_idx = np.where(img > 127.5)
        img_rgb[r_idx, c_idx, :] = color_R
        r_idx, c_idx = np.where(img < 127.5)
        img_rgb[r_idx, c_idx, :] = color_G
        scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_R_G.png'%(i,label)))
    else:
        if rand_num < 1.0/6.0:
            ## Foreground: R, Background:G
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_R
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_G
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_R_G.png'%(i,label)))
        elif rand_num < 2.0/6.0:
            ## Foreground: G, Background:R
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_G
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_R
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_G_R.png'%(i,label)))
        elif rand_num < 3.0/6.0:
            ## Foreground: R, Background:B
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_R
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_B
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_R_B.png'%(i,label)))
        elif rand_num < 4.0/6.0:
            ## Foreground: B, Background:R
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_B
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_R
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_B_R.png'%(i,label)))
        elif rand_num < 5.0/6.0:
            ## Foreground: G, Background:B
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_G
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_B
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_G_B.png'%(i,label)))
        else:
            ## Foreground: B, Background:G
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_B
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_G
            scipy.misc.toimage(img_rgb).save(os.path.join(B_dir, '%06d_%d_B_G.png'%(i,label)))

if not IsTrain:
    ## test_A2B
    for i in tqdm.tqdm(range(data_images.shape[0])):
    # for i in range(1):
        img = data_images[idx_list_A[i],:,:]
        label = data_labels[idx_list_A[i]]
        img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
        # rand_num = np.random.uniform(0.0,1.0)
        rand_num = rand_num_list_B[i]
        if IsTrain and label > 4:  ## In order to show the model can also transfer the style of unseen samples
            ## Foreground: R, Background:G
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_R
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_G
            scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_R_G.png'%(i,label)))
        else:
            if rand_num < 1.0/6.0:
                ## Foreground: R, Background:G
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_R
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_G
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_R_G.png'%(i,label)))
            elif rand_num < 2.0/6.0:
                ## Foreground: G, Background:R
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_G
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_R
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_G_R.png'%(i,label)))
            elif rand_num < 3.0/6.0:
                ## Foreground: R, Background:B
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_R
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_B
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_R_B.png'%(i,label)))
            elif rand_num < 4.0/6.0:
                ## Foreground: B, Background:R
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_B
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_R
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_B_R.png'%(i,label)))
            elif rand_num < 5.0/6.0:
                ## Foreground: G, Background:B
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_G
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_B
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_G_B.png'%(i,label)))
            else:
                ## Foreground: B, Background:G
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_B
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_G
                scipy.misc.toimage(img_rgb).save(os.path.join(ab_dir, '%06d_%d_B_G.png'%(i,label)))

    ## test_B2A
    for i in tqdm.tqdm(range(data_images.shape[0])):
    # for i in range(1):
        img = data_images[idx_list_B[i],:,:]
        label = data_labels[idx_list_B[i]]
        img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
        # rand_num = np.random.uniform(0,1)
        rand_num = rand_num_list_A[i]
        if rand_num < 0.5:
            # pdb.set_trace()
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_Black
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_White
            scipy.misc.toimage(img_rgb).save(os.path.join(ba_dir, '%06d_%d_Black_Whilte.png'%(i,label)))
        else:
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_White
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_Black
            scipy.misc.toimage(img_rgb).save(os.path.join(ba_dir, '%06d_%d_Whilte_Black.png'%(i,label)))