data_dir='/esat/dragon/liqianma/datasets/Adaptation/cityscapes/leftImg8bit/all_train_extra/'

## ColorFrontCenter
# out_dir='/Storage/Liqian/datasets/Carla/20180829_ZPG_ColorFrontCenter_512x256/'
# mkdir -p ${out_dir}
# for img_path in ${data_dir}color-*_front_center.png; do
#     convert ${img_path} -resize 512x256\> ${out_dir}${img_path##*/}
# done

## ColorLeftWide
out_dir='/esat/dragon/liqianma/datasets/Adaptation/cityscapes/leftImg8bit/all_train_extra_512x256/'
mkdir -p ${out_dir}
for img_path in ${data_dir}*.png; do
    convert ${img_path} -resize 512x256\> ${out_dir}${img_path##*/}
done
