from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from easydict import EasyDict as edict

cfg = edict()

cub200 = edict()
cub200._RGB_MEAN = [124.41, 128.99, 114.18]
cub200.img_dir = "/data/CUB_200_2011/CUB_200_2011/images"
cub200.img_annot_for_train = "/data/CUB_200_2011/CUB_200_2011/train.pkl"
cub200.img_annot_for_test = "/data/CUB_200_2011/CUB_200_2011/test.pkl"

aircraft = edict()
aircraft._RGB_MEAN = [121.2417, 128.7999, 134.2618]
aircraft.img_dir = "/data/fgvc-aircraft-2013b/data/images"
aircraft.img_annot_for_train = "/data/fgvc-aircraft-2013b/data/images_variant_trainval.pkl"
aircraft.img_annot_for_test = "/data/fgvc-aircraft-2013b/data/images_variant_test.pkl"

standfordcars = edict()
standfordcars._RGB_MEAN = [119.77, 116.96, 115.54]
standfordcars.img_dir_for_train = "/data/standford-cars/cars_train/"
standfordcars.img_dir_for_test = "/data/standford-cars/cars_test/"
standfordcars.img_annot_for_train = "/data/standford-cars/devkit/cars_train_annos.pkl"
standfordcars.img_annot_for_test = "/data/standford-cars/devkit/cars_test_annos.pkl"


cfg.PRINT_LOSS_INTERVAL = 20
cfg.RUN_SUMMARY_INTERVAL = 100
cfg.RESTORE_INTERVAL = 1500

# Choose the dataset you want to train/test
cfg.current_dataset = "aircraft"

if cfg.current_dataset == "cub200":
    cfg._RGB_MEAN = cub200._RGB_MEAN
    cfg.num_classes = 200

elif cfg.current_dataset == "aircraft":
    cfg._RGB_MEAN = aircraft._RGB_MEAN
    cfg.num_classes = 100

elif cfg.current_dataset == "standfordcars":
    cfg._RGB_MEAN = standfordcars._RGB_MEAN
    cfg.num_classes = 196

else:
    raise ValueError('Name of dataset unknown %s' % cfg.current_dataset)

cfg.log_dir_stage1 = "log/"+cfg.current_dataset+"_bcnn_stage1"
cfg.log_dir_stage2 = "log/"+cfg.current_dataset+"_bcnn_stage2"

cfg.train_dir_stage1 = "/data/" + cfg.current_dataset + "_bcnn_stage1"
cfg.train_dir_stage2 = "/data/" + cfg.current_dataset + "_bcnn_stage2"

cfg.vgg_pretrained_path = "/data/model/tf_model/vgg_16.ckpt"
