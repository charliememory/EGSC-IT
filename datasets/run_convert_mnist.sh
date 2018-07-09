source ~/.bashrc_liqianma

for IsTrain in 0 1
do
    ##
    data_dir='../data/mnist_BW'
    python prepare_data_mnist_BW.py ${data_dir} ${IsTrain}
    python convert_mnist_BW.py ${data_dir} ${IsTrain}
    ##
    data_dir='../data/mnist_multi_jitterColor_BW'
    python prepare_data_mnist_multiJitter_BW.py ${data_dir} ${IsTrain}
    python convert_mnist_multi_BW.py ${data_dir} ${IsTrain}
done
