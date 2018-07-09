import mnist, os, sys, pdb, tqdm
import scipy.misc
import numpy as np
import colorsys

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

DIGIT_NUM = 10
SQUARE_SIZE = 4
img_size = 28
jitter = 0.4
color_Black = [0,0,0]
color_White = [255,255,255]
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
idx_list_list_A = []
square_idx_list_list_A = []
for i in tqdm.tqdm(range(data_images.shape[0])):
    sample_idx_list_tmp = list((np.random.rand(100)*data_images.shape[0]).astype(np.int))
    idx_dic = {}
    for idx in sample_idx_list_tmp:
        label = data_labels[idx]
        if label not in idx_dic:
            idx_dic[label] = idx
        if len(idx_dic) == DIGIT_NUM:
            break
    idx_list = np.random.permutation(list(idx_dic.values()))
    idx_list_list_A.append(idx_list)

    square_idx_list = np.random.permutation(SQUARE_SIZE*SQUARE_SIZE)
    square_idx_list_list_A.append(square_idx_list)
    rand_num = rand_num_list_A[i]
    # comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
    if rand_num < 0.5:
        comb_img = np.ones([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])*255
    else:
        comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
    # rand_num = np.random.uniform(0.0,1.0)
    for j in range(len(idx_list)):
        img = data_images[idx_list[j],:,:] # * -1 + 256
        img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
        if rand_num < 0.5:
            # pdb.set_trace()
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_Black
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_White
        else:
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color_White
            r_idx, c_idx = np.where(img < 127.5)
            img_rgb[r_idx, c_idx, :] = color_Black
        rr, cc = square_idx_list[j]/SQUARE_SIZE, square_idx_list[j]%SQUARE_SIZE
        comb_img[rr*img_size:(rr+1)*img_size, cc*img_size:(cc+1)*img_size, :] = img_rgb
    if rand_num < 0.5:
        scipy.misc.toimage(comb_img).save(os.path.join(A_dir, '%06d_Black_Whilte.png'%i))
    else:
        scipy.misc.toimage(comb_img).save(os.path.join(A_dir, '%06d_Whilte_Black.png'%i))


## train_B
idx_list_list_B = []
square_idx_list_list_B = []
color_dic_list_B = []
for i in tqdm.tqdm(range(data_images.shape[0])):
    sample_idx_list_tmp = list((np.random.rand(200)*data_images.shape[0]).astype(np.int))
    idx_dic = {}
    for idx in sample_idx_list_tmp:
        label = data_labels[idx]
        if label not in idx_dic:
            idx_dic[label] = idx
        if len(idx_dic) == DIGIT_NUM:
            break
    idx_list = np.random.permutation(list(idx_dic.values()))
    idx_list_list_B.append(idx_list)

    hsv_list = []
    for j in range(DIGIT_NUM):
        hsv_list.append([float(j)/DIGIT_NUM, 0.5, 0.5])
    square_idx_list = np.random.permutation(SQUARE_SIZE*SQUARE_SIZE)
    square_idx_list_list_B.append(square_idx_list)
    # comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
    rand_num = rand_num_list_B[i]
    if rand_num < 0.5:
        comb_img = np.ones([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])*255
    else:
        comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
    color_dic = {}
    for j in range(len(idx_list)):
        img = data_images[idx_list[j],:,:] # * -1 + 256
        label = data_labels[idx_list[j]]
        img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
        hsv = hsv_list[label]
        color = hsv2rgb(hsv[0], hsv[1]*np.random.uniform(1.-jitter, 1.+jitter), hsv[2]*np.random.uniform(1.-jitter, 1.+jitter))
        color_dic[label] = color
        # pdb.set_trace()
        r_idx, c_idx = np.where(img > 127.5)
        img_rgb[r_idx, c_idx, :] = color
        r_idx, c_idx = np.where(img < 127.5)
        if rand_num < 0.5:
            img_rgb[r_idx, c_idx, :] = color_White
        else:
            img_rgb[r_idx, c_idx, :] = color_Black
        rr, cc = square_idx_list[j]/SQUARE_SIZE, square_idx_list[j]%SQUARE_SIZE
        comb_img[rr*img_size:(rr+1)*img_size, cc*img_size:(cc+1)*img_size, :] = img_rgb
    color_dic_list_B.append(color_dic)
    scipy.misc.toimage(comb_img).save(os.path.join(B_dir, '%06d.png'%i))


if not IsTrain:
    ## train_A2B
    for i in tqdm.tqdm(range(data_images.shape[0])):
        idx_list = idx_list_list_A[i]
        # hsv_list = []
        # for j in range(DIGIT_NUM):
        #     hsv_list.append([float(j)/DIGIT_NUM, 0.5, 0.5])
        # square_idx_list = np.random.permutation(SQUARE_SIZE*SQUARE_SIZE)
        square_idx_list = square_idx_list_list_A[i]
        # comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
        rand_num = rand_num_list_B[i]
        if rand_num < 0.5:
            comb_img = np.ones([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])*255
        else:
            comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
        color_dic = color_dic_list_B[i]
        for j in range(len(idx_list)):
            img = data_images[idx_list[j],:,:] # * -1 + 256
            label = data_labels[idx_list[j]]
            img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
            color = color_dic[label]
            # hsv = hsv_list[label]
            # color = hsv2rgb(hsv[0], hsv[1]*np.random.uniform(1.-jitter, 1.+jitter), hsv[2]*np.random.uniform(1.-jitter, 1.+jitter))
            # color_dic[label] = color
            # pdb.set_trace()
            r_idx, c_idx = np.where(img > 127.5)
            img_rgb[r_idx, c_idx, :] = color
            r_idx, c_idx = np.where(img < 127.5)
            if rand_num < 0.5:
                img_rgb[r_idx, c_idx, :] = color_White
            else:
                img_rgb[r_idx, c_idx, :] = color_Black
            rr, cc = square_idx_list[j]/SQUARE_SIZE, square_idx_list[j]%SQUARE_SIZE
            comb_img[rr*img_size:(rr+1)*img_size, cc*img_size:(cc+1)*img_size, :] = img_rgb
        # color_dic_list.append(color_dic)
        scipy.misc.toimage(comb_img).save(os.path.join(ab_dir, '%06d.png'%i))


    ## train_B2A
    for i in tqdm.tqdm(range(data_images.shape[0])):
        idx_list = idx_list_list_B[i]
        # square_idx_list = np.random.permutation(SQUARE_SIZE*SQUARE_SIZE)
        square_idx_list = square_idx_list_list_B[i]
        # comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
        # rand_num = np.random.uniform(0.0,1.0)
        rand_num = rand_num_list_A[i]
        if rand_num < 0.5:
            comb_img = np.ones([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])*255
        else:
            comb_img = np.zeros([SQUARE_SIZE*img_size, SQUARE_SIZE*img_size, 3])
        for j in range(len(idx_list)):
            img = data_images[idx_list[j],:,:] # * -1 + 256
            img_rgb = np.tile(np.expand_dims(img, -1), [1,1,3])
            if rand_num < 0.5:
                # pdb.set_trace()
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_Black
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_White
            else:
                r_idx, c_idx = np.where(img > 127.5)
                img_rgb[r_idx, c_idx, :] = color_White
                r_idx, c_idx = np.where(img < 127.5)
                img_rgb[r_idx, c_idx, :] = color_Black
            rr, cc = square_idx_list[j]/SQUARE_SIZE, square_idx_list[j]%SQUARE_SIZE
            comb_img[rr*img_size:(rr+1)*img_size, cc*img_size:(cc+1)*img_size, :] = img_rgb
        if rand_num < 0.5:
            scipy.misc.toimage(comb_img).save(os.path.join(ba_dir, '%06d_Black_Whilte.png'%i))
        else:
            scipy.misc.toimage(comb_img).save(os.path.join(ba_dir, '%06d_Whilte_Black.png'%i))







