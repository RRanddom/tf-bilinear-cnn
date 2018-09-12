from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from easydict import EasyDict as edict

cfg = edict()

# Download CUB-200-2011 at [http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz]
cub200 = edict()
cub200._RGB_MEAN = [124.41, 128.99, 114.18]
cub200.img_dir = "/data/CUB_200_2011/CUB_200_2011/images"
cub200.img_annot_for_train = "../data/cub-200/train.pkl"
cub200.img_annot_for_test = "../data/cub-200/test.pkl"

# Download 
aircraft = edict()
aircraft._RGB_MEAN = [121.2417, 128.7999, 134.2618]
aircraft.img_dir = "/data/fgvc-aircraft-2013b/data/images"
aircraft.img_annot_for_train = "../data/images_variant_trainval.pkl"
aircraft.img_annot_for_test = "../data/images_variant_test.pkl"

# Download car dataset at [http://imagenet.stanford.edu/internal/car196/cars_train.tgz, http://imagenet.stanford.edu/internal/car196/cars_test.tgz, http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz]
standfordcars = edict()
standfordcars._RGB_MEAN = [119.77, 116.96, 115.54]
standfordcars.img_dir_for_train = "/data/standford-cars/cars_train/"
standfordcars.img_dir_for_test = "/data/standford-cars/cars_test/"
standfordcars.img_annot_for_train = "../data/cars_train_annots.pkl"
standfordcars.img_annot_for_test = "../data/cars_test_annots.pkl"

cfg.PRINT_LOSS_INTERVAL = 20
cfg.RUN_SUMMARY_INTERVAL = 100
cfg.RESTORE_INTERVAL = 1500

cfg.stage1_base_lr = .9
cfg.stage2_base_lr = 1e-3
cfg.momentum = .9
cfg.BATCH_SIZE = 32

# Choose the dataset you want to train/test
cfg.current_dataset = "cub200" # one of ["cub200", "aircraft", "standfordcars"]

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

# the TensorFlow event files will be placed under log/DATASET_NAME_****/. You can change the output directory
# cfg.log_dir_stage1 = "log/"+cfg.current_dataset+"_bcnn_stage1"
# cfg.log_dir_stage2 = "log/"+cfg.current_dataset+"_bcnn_stage2"

# the TensorFlow checkpoint file will be placed under /data/DATASET_NAME_****/. You can change the output directory
# cfg.train_dir_stage1 = "/data/" + cfg.current_dataset + "_bcnn_stage1"
# cfg.train_dir_stage2 = "/data/" + cfg.current_dataset + "_bcnn_stage2"


cfg.train_dir = '/data/' + cfg.current_dataset + 'train_dir'

# Download pretrained vgg model at [http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz]
cfg.vgg_pretrained_path = "/data/model/tf_model/vgg_16.ckpt"
