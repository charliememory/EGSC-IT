source ~/.bashrc_liqianma

#### Download GTA5 and Cityscape datasets, then uncompress them into '../data/'

## Define data path
src_A_dir='../data/GTA5'
src_B_dir='../data/cityscapes'
dst_dir='../data/gta25k_city'
# rm -rf ${dst_dir}
mkdir ${dst_dir}
mkdir ${dst_dir}'/testA'
mkdir ${dst_dir}'/testB'
mkdir ${dst_dir}'/testA_seg'
mkdir ${dst_dir}'/testB_seg'
mkdir ${dst_dir}'/trainA'
mkdir ${dst_dir}'/trainB'
mkdir ${dst_dir}'/trainA_seg'
mkdir ${dst_dir}'/trainB_seg'
mkdir ${dst_dir}'/trainA_seg_class'
mkdir ${dst_dir}'/trainB_seg_class'
ln -s ${src_B_dir}'/leftImg8bit/all_train_extra' ${dst_dir}'/trainB_extra'

## Process src data and define specific path
## For cityscapes train
cd ${src_B_dir}'/leftImg8bit'
mkdir 'all_train/'
cp `find train/ -name "*.png"` 'all_train/'
cd -
cd ${src_B_dir}'/gtFine'
mkdir 'all_train/'
cp `find train/ -name "*.png"` 'all_train/'
cd -
# For cityscapes val
cd ${src_B_dir}'/leftImg8bit'
mkdir 'all_val/'
cp `find val/ -name "*.png"` 'all_val/'
cd -
cd ${src_B_dir}'/gtFine'
mkdir 'all_val/'
cp `find val/ -name "*.png"` 'all_val/'
cd -

## Excute prepare_data
A_imagepath=${src_A_dir}'/images/'
A_segpath=${src_A_dir}'/labels/'
B_imagepath=${src_B_dir}'/leftImg8bit/all_val/'
B_segpath=${src_B_dir}'/gtFine/all_val/'
python prepare_data_gta_city.py --A_imagepath=${A_imagepath} \
                    --A_segpath=${A_segpath} \
                    --B_imagepath=${B_imagepath} \
                    --B_segpath=${B_segpath}

## Generate segment class
python segment_class.py --data_dir=${dst_dir} --dataset='gta25k' --use_8catId=True

## Conver to tf-record
for IsTrain in 0 1
do
    python convert_gta.py ${dst_dir} ${IsTrain}
done