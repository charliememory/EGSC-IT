# Exemplar Guided Unsupervised Image-to-Image
Tensorflow implementation of NIPS 2018 submission [Exemplar Guided Unsupervised Image-to-Image](https://arxiv.org/abs/1805.11145)

![alt text](imgs/teaser_GTA2BDD.svg)

## Network architecture
    ![alt text](imgs/framework_EGUNIT_test.svg)
    ![alt text](imgs/info_flow_in_autoencoder.png)

## Dependencies
    - python 2.7
    - tensorflow-gpu (1.4.1)
    - numpy (1.14.0)
    - Pillow (5.0.0)
    - scikit-image (0.13.0)
    - scipy (1.0.1)
    - matplotlib (2.0.0)


## Resources
    - Pretrained models: [MNIST](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/mnist_models.zip), [MNIST_multi](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/mnist_multi_models.zip), [GTA<->Cityscapes](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/gta_city_models.zip), [GTA<->BDD](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/gta_bdd_models.zip), [GTA<->Cityscapes](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/gta_city_models.zip), [CelebA](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/models/celeba_models.zip)
    - Training & Testing data in tf-record format: [MNIST](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/mnist_tf.zip), [MNIST_multi](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/mnist_multi_tf.zip), [GTA<->Cityscapes](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/gta_city_tf.zip). [GTA<->BDD](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/gta_bdd_tf.zip), [CelebA](http://homes.esat.kuleuven.be/~liqianma/NIPS18_EGUNIT/data/celeba_tf.zip).
    Note: For the GTA<->Cityscapes and GTA<->BDD experiments, the data are prepared with RGB images of 512x1024 resolution, and segmentation labels of 8 categories. They are provided used for further research. In our paper, we use RGB images of 256x512 resolution without and segmentation labels.
    - Segmentation model
    Refer to [DeepLab-ResNet-TensorFlow](https://github.com/DrSleep/tensorflow-deeplab-resnet)

## TF-record data preparation steps
    You can skip this data preparation procedure if directly using the tf-record data files.
    1. `cd datasets`
    2. `./run_convert_mnist.sh` to download and convert mnist and mnist_multi to tf-record format.
    3. `./run_convert_gta_city.sh` to convert the images and segmentation to tf-record format. You need to download data from [GTA5 website](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Cityscapes website](https://www.cityscapes-dataset.com/)
    4. `./run_convert_gta_bdd.sh` to convert the images and segmentation to tf-record format. You need to download data from [BDD website](http://bdd-data.berkeley.edu/). Note: this script will reuse gta data downloaded and processed in `./run_convert_gta_city.sh`
    5. `./run_convert_celeba.sh` to convert the images to tf-record format. You can directly download the prepared data or download and process data from [CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) .

## Training steps
    1. Replace the links `data`, 'logs', 'weights' with your own directories or links.
    2. Download the tf-record training data to the `data_parent_dir` (default `./data`).
    3. Modify the `data_parent_dir`, `checkpoint_dir` and comment/uncomment the target experiment in the `run_train_feaMask.sh` and `run_train_EGUNIT.sh` scripts.
    4. run `run_train_feaMask.sh` and `run_train_EGUNIT.sh`.
 
## Testing steps
    1. Replace the links `data`, 'logs', 'weights' with your own directories or links.
    2. (Optional) Download the pretrained models to the `checkpoint_dir` (default `./logs`).
    3. Download the tf-record testing data to the `data_parent_dir` (default `./data`).
    4. Modify the `data_parent_dir`, `checkpoint_dir` and comment/uncomment the target experiment in the `run_test_EGUNIT.sh` script.
    5. run `run_test_EGUNIT.sh`. 

## Citation
```
@article{ma2018exemplar,
  title={Exemplar Guided Unsupervised Image-to-Image Translation},
  author={Ma, Liqian and Jia, Xu and Georgoulis, Stamatios and Tuytelaars, Tinne and Van Gool, Luc},
  journal={arXiv preprint arXiv:1805.11145},
  year={2018}
}
```

## Related projects
- [UNIT-Tensorflow](https://github.com/taki0112/UNIT-Tensorflow)
- [SG-GAN](https://github.com/Peilun-Li/SG-GAN)
- [Pose-Guided-Person-Image-Generation](https://github.com/charliememory/Pose-Guided-Person-Image-Generation)


## Results