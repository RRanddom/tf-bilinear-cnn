from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os 
import numpy as np
import tensorflow as tf

from data.dataset_factory import get_dataset
from tools.config import cfg
from model.bilinear_cnn import BilinearCnn
from model.vgg import Vgg

def snapshot(sess, saver, _iter, train_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    file_name = "bilinear"+"_"+str(_iter)+".ckpt"
    full_path = os.path.join(train_dir, file_name)
    saver.save(sess, full_path)
    print ("write snapshot to:",full_path)


def train_for_stage1(pretrained_vgg_path, log_dir, train_dir, learning_rate=0.9):

    bcnn = BilinearCnn(is_training=True, fine_tuning=False, num_class=cfg.num_classes)
    loss = bcnn.loss

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()

    all_vars = tf.global_variables()
    vars_from_vgg = [var for var in all_vars if var.name.startswith("vgg_16")]

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(vars_from_vgg)
    saver.restore(sess, pretrained_vgg_path)
    
    print ("init done")

    writer = tf.summary.FileWriter(log_dir, sess.graph)
    data_layer = get_dataset(name=cfg.current_dataset, mode="TRAIN")
    saver = tf.train.Saver(max_to_keep=3)

    _iter = 1
    for blob in data_layer.data_iterator(num_epochs=40):
        if _iter % cfg.RUN_SUMMARY_INTERVAL == 0:
            loss_value, summaries, _ = bcnn.train_step_with_summary(sess, blob, train_op)
            writer.add_summary(summaries, _iter)
        else :
            loss_value, _ = bcnn.train_step(sess, blob, train_op)

        if _iter % cfg.PRINT_LOSS_INTERVAL == 0:
            print ("iteration:{} loss:{}".format(_iter, loss_value))
        
        if _iter % cfg.RESTORE_INTERVAL == 0:
            snapshot(sess, saver, _iter, train_dir)

        _iter += 1
        
    snapshot(sess, saver, _iter, train_dir)
    writer.close()

    return tf.train.latest_checkpoint(train_dir)

def train_for_stage2(ckpt_path, log_dir, train_dir, learning_rate):
    
    bcnn = BilinearCnn(is_training=True, fine_tuning=True, num_class=cfg.num_classes)
    loss = bcnn.loss

    all_vars = tf.global_variables()

    vars_we_should_restore = [var for var in all_vars if var.name.startswith("vgg_16") or var.name.startswith("fc-layer")]

    sess = tf.Session()
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=vars_we_should_restore)
    saver.restore(sess, ckpt_path)

    writer = tf.summary.FileWriter(log_dir, sess.graph)
    data_layer = get_dataset(name=cfg.current_dataset, mode="TRAIN")
    saver = tf.train.Saver(max_to_keep=3)
    
    _iter = 1
    for blob in data_layer.data_iterator(num_epochs=25):
        if _iter % cfg.RUN_SUMMARY_INTERVAL == 0:
            loss_, summaries, _ = bcnn.train_step_with_summary(sess, blob, train_op)
            writer.add_summary(summaries, _iter)

        loss_, _ = bcnn.train_step(sess, blob, train_op)
        
        if _iter % cfg.PRINT_LOSS_INTERVAL == 0:
            print ("iteration:{} loss:{}".format(_iter, loss_))
        
        if _iter % cfg.RESTORE_INTERVAL == 0:
            snapshot(sess, saver, _iter, train_dir)

        _iter += 1

    snapshot(sess, saver, _iter, train_dir)
    writer.close()

if __name__ == "__main__":

    vgg_path = cfg.vgg_pretrained_path

    log_dir_stage1 = cfg.log_dir_stage1
    log_dir_stage2 = cfg.log_dir_stage2

    train_dir_stage1 = cfg.train_dir_stage1
    train_dir_stage2 = cfg.train_dir_stage2

    stage1_lr = .9
    stage2_lr = 1e-3
    
    restore_path = train_for_stage1(pretrained_vgg_path=vgg_path, log_dir=log_dir_stage1, train_dir=train_dir_stage1, learning_rate=stage1_lr)
    tf.reset_default_graph()
    train_for_stage2(ckpt_path=restore_path, log_dir=log_dir_stage2, train_dir=train_dir_stage2, learning_rate=stage2_lr)
