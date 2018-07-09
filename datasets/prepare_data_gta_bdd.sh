source ~/.bashrc_liqianma

## Define data path
src_A_dir='/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/gta25k'
src_B_dir='/esat/dragon/liqianma/datasets/Adaptation/BDD'
dst_dir='/esat/dragon/liqianma/datasets/Adaptation/SG-GAN_data/gta25k_bdd'
link_dir='/users/visics/liqianma/workspace/SG-GAN/datasets/'
# rm -rf ${dst_dir}
mkdir ${dst_dir}
ln -s ${dst_dir} ${link_dir}
ln -s ${src_A_dir}'/trainA' ${dst_dir}'/trainA'
ln -s ${src_A_dir}'/trainA_seg' ${dst_dir}'/trainA_seg'
ln -s ${src_A_dir}'/trainA_seg_class_8catId' ${dst_dir}'/trainA_seg_class_8catId'
ln -s ${src_A_dir}'/trainA_seg_class_20trainId' ${dst_dir}'/trainA_seg_class_20trainId'
ln -s ${src_A_dir}'/testA' ${dst_dir}'/testA'
ln -s ${src_A_dir}'/testA_seg' ${dst_dir}'/testA_seg'
ln -s ${src_A_dir}'/testA_seg_class_8catId' ${dst_dir}'/testA_seg_class_8catId'
ln -s ${src_A_dir}'/testA_seg_class_20trainId' ${dst_dir}'/testA_seg_class_20trainId'
ln -s ${src_B_dir}'/segmentation/train/raw_images' ${dst_dir}'/trainB'
ln -s ${src_B_dir}'/segmentation/train/class_color' ${dst_dir}'/trainB_seg'

ln -s ${src_B_dir}'/videos/out_img' ${dst_dir}'/trainB_extra'

ln -s ${src_B_dir}'/segmentation/val/raw_images' ${dst_dir}'/testB'
ln -s ${src_B_dir}'/segmentation/val/class_color' ${dst_dir}'/testB_seg'

## Generate segment class
python segment_class.py --dataset='gta25k_bdd' --use_8catId=False
# python segment_class.py --dataset='gta25k_bdd' --use_8catId=True
 