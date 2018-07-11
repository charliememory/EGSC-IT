cd /esat/diamond/liqianma/HomepageResources/NIPS18_EGUNIT/data_tmp
# zip -r celeba_tf.zip celeba/celebaMaleFemale_test_128x128 celeba/celebaMaleFemale_train_128x128
# zip -r gta_bdd_tf.zip gta_bdd/gta25k_bdd_test_512x1024_8catId gta_bdd/gta25k_bdd_train_512x1024_8catId
# zip -r gta_city_tf.zip gta_city/gta25k_city_test_512x1024_8catId gta_city/gta25k_city_train_512x1024_8catId
zip -r mnist_tf.zip mnist_BW_test_28x28 mnist_BW_train_28x28
zip -r mnist_multi_tf.zip mnist_multi_jitterColor_BW_test_112x112 mnist_multi_jitterColor_BW_train_112x112

cd /esat/diamond/liqianma/HomepageResources/NIPS18_EGUNIT/models
zip -r celeba_models.zip MODEL0_celeba_bs8_lr1e-4 MODEL1_celeba_bs8_lr1e-4_5e3Style_1e1Content
zip -r gta_bdd_models.zip MODEL0_gta25kbdd_bs3_lr1e-4 MODEL0_gta25kbdd_bs3_lr1e-4
zip -r gta_city_models.zip MODEL0_gta25kcity_bs3_lr1e-4 MODEL0_gta25kcity_bs3_lr1e-4
zip -r mnist_models.zip MODEL0_mnist_bs8_lr1e-5 MODEL1_mnist_bs8_lr1e-5_1e3Style_1e1Content
zip -r mnist_multi_models.zip MODEL0_mnist_multi_bs8_lr1e-5 MODEL1_mnist_multi_bs8_lr1e-5_1e4Style_1e2Content