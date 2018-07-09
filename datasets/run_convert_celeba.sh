source ~/.bashrc_liqianma

## Define data path
src_dir='../data/celeba'
dst_dir='../data/celeba'

if [ ! -d ../data/celeba/testB ]; then
    #### Option 1: Download prepared data, then skip Option 2
    # cd ../data
    # wget homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/celeba/celeba_crop_resize_tr_ts.zip
    # unzip celeba_crop_resize_tr_ts.zip
    # mv celeba_crop_resize_tr_ts celeba
    # rm -f celeba_crop_resize_tr_ts.zip
    # cd -

    #### Option 2: Download raw data and divide data manually from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    ## Step 1: download and uncompress 'Align&Cropped Images', 'Attributes Annotations' and 'Train/Val/Test Partitions' to ../data/celeba
    ## Step 2: use script to divide male & female images, and also do crop & resize
    python divide_celeba.py  ${src_dir}

    # rm -rf ${dst_dir}
    mkdir ${dst_dir}
    ln -s ${src_dir}'/tr_male_img' ${dst_dir}'/trainA'
    ln -s ${src_dir}'/tr_female_img' ${dst_dir}'/trainB'
    ln -s ${src_dir}'/ts_male_img' ${dst_dir}'/testA'
    ln -s ${src_dir}'/ts_female_img' ${dst_dir}'/testB'
fi

for IsTrain in 0 1
do
    python convert_celeba.py ${dst_dir} ${IsTrain}
done