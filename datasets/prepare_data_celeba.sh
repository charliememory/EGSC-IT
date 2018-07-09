source ~/.bashrc_liqianma

## Define data path
src_dir='/esat/dragon/liqianma/datasets/Adaptation/Celeba'
dst_dir='/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/celebaMaleFemale_crop_resize'
link_dir='/users/visics/liqianma/workspace/SG-GAN/datasets/'
# rm -rf ${dst_dir}
mkdir ${dst_dir}
ln -s ${dst_dir} ${link_dir}
ln -s ${src_dir}'/tr_male_img' ${dst_dir}'/trainA'
ln -s ${src_dir}'/tr_female_img' ${dst_dir}'/trainB'
ln -s ${src_dir}'/ts_male_img' ${dst_dir}'/testA'
ln -s ${src_dir}'/ts_female_img' ${dst_dir}'/testB'