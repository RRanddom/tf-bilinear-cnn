from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os 
import pickle

from model.bilinear_cnn import BilinearCnn
from data.dataset_factory import get_dataset
from tools.config import cfg

if __name__ == "__main__":

    ckpt_path = cfg.train_dir_stage2

    dataset = get_dataset(name=cfg.current_dataset, mode="TEST")

    bcnn = BilinearCnn(is_training=False, fine_tuning=False, num_class=cfg.num_classes)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

    print ("save done")

    total_right_ct = 0
    total_wrong_ct = 0

    iter_ct = 1
    # gt_labels = []
    # predicts = []

    for blob in dataset.data_iterator(num_epochs=1):
        predict = bcnn.test_step(sess, blob)
        
        iter_ct += 1

        total_right_ct += np.sum(predict == blob["cls"])
        total_wrong_ct += np.sum(predict != blob["cls"])
        
        # gt_labels.extend(list(blob["cls"]))
        
        print ("total right ct:{} total wrong ct:{}".format(total_right_ct, total_wrong_ct))
    print ("overall accuracy:{}".format(total_right_ct/(total_right_ct + total_wrong_ct)))
