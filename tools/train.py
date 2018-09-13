from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os 
import numpy as np
import tensorflow as tf

from data.dataset_factory import get_dataset
from tools.config import cfg
from model.bilinear_cnn import bilinear_cnn

def _preprocess_for_training(input_image, input_height, input_width, image_name, image_label, label_desc):
    input_image = tf.expand_dims(input_image, 0)
    input_image = tf.cast(input_image, dtype=tf.float32)

    _R_MEAN, _G_MEAN, _B_MEAN = cfg._RGB_MEAN
    rgb_mean = tf.reshape(np.array([_R_MEAN, _G_MEAN, _B_MEAN]).astype(np.float32), [1,1,1,3])
    input_image = input_image - rgb_mean    

    resized = tf.image.resize_images(input_image, (488, 488))
    crop_fn = lambda x: tf.random_crop(x, [448, 448, 3])
    processed = tf.map_fn(crop_fn, resized)
    flip_fn = lambda x: tf.image.random_flip_left_right(x)
    processed = tf.map_fn(flip_fn, processed)
    brightness_fn = lambda x: tf.image.random_brightness(x, max_delta=0.2)
    processed = tf.map_fn(brightness_fn, processed)

    processed = processed[0]
    return processed, input_height, input_width, image_name, image_label, label_desc,


def input_pipeline(num_epochs=10):

    dataset = get_dataset(dataset_name=cfg.current_dataset, split_name='train')
    dataset = dataset.map(_preprocess_for_training)
    dataset = dataset.shuffle(buffer_size=500).repeat(num_epochs).batch(cfg.BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    input_image, input_height, input_width, image_name, image_label, label_desc = iterator.get_next()
    return input_image, image_label


def get_init_fn_for_train(vgg_pretrained_path, model_dir, exclude_vars=[]):

    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info("Ignore pretrained vgg model because a checkpoint file already exists")
        return None
    
    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if 'vgg_16' in var.name:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(vgg_pretrained_path):
        vgg_pretrained_path = tf.train.latest_checkpoint(vgg_pretrained_path)
    
    tf.logging.info("Fine tuning from %s" %(vgg_pretrained_path))

    if not variables_to_restore:
        raise ValueError("variables to restore cannot be empty.")
    
    saver = tf.train.Saver(variables_to_restore, reshape=False)
    saver.build()

    def callback(scaffold, session):
        saver.restore(session, vgg_pretrained_path)

    return callback

def get_init_fn_for_finetune(train_dir, finetune_dir):

    if tf.train.latest_checkpoint(finetune_dir):
        tf.logging.info("Ignore pretrained model because a checkpoint file already exists")
        return None

    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if not 'Momentum' in var.name:
            variables_to_restore.append(var)
            
    saver = tf.train.Saver(variables_to_restore)
    saver.build()

    if tf.gfile.IsDirectory(train_dir):
        train_dir = tf.train.latest_checkpoint(train_dir)

    def callback(scaffold, sess):
        saver.restore(sess, train_dir)

    return callback


def bcnn_train_model(features, labels, mode, params):
    """
    Args:
        features: input image batch
        labels:   image label
        mode: 'TRAIN' | 'EVAL' | 'PREDICT'
        params: additional params
    """
    logits = bilinear_cnn(features, is_training=True, fine_tuning=False, num_class=cfg.num_classes)
    optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.train_base_lr, momentum=cfg.momentum)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=cfg.num_classes), logits=logits)
    tf.summary.scalar('softmax_loss', loss)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step)

    return tf.estimator.EstimatorSpec(mode=mode, 
                                      predictions=logits, 
                                      loss=loss, 
                                      train_op=train_op,
                                      scaffold=tf.train.Scaffold(init_fn=get_init_fn_for_train(cfg.vgg_pretrained_path, cfg.train_dir)))


def bcnn_finetune_model(features, labels, mode, params):
    '''
        features: input image batch
        labels:   image label
        mode: 'TRAIN' | 'EVAL' | 'PREDICT'
        params: additional params
    '''
    logits = bilinear_cnn(features, is_training=True, fine_tuning=True, num_class=cfg.num_classes)
    optimizer = tf.train.MomentumOptimizer(learning_rate=cfg.finetune_base_lr, momentum=cfg.momentum)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=cfg.num_classes), logits=logits)
    tf.summary.scalar('softmax_loss', loss)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss, global_step)

    return tf.estimator.EstimatorSpec(mode=mode, 
                                      predictions=logits, 
                                      loss=loss, 
                                      train_op=train_op,
                                      scaffold=tf.train.Scaffold(init_fn=get_init_fn_for_finetune(cfg.train_dir, cfg.finetune_dir)))


def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    run_config = tf.estimator.RunConfig()\
                    .replace(save_summary_steps=2500)\
                    .replace(log_step_count_steps=10)

    train_dir = cfg.train_dir
    model = tf.estimator.Estimator(model_fn=bcnn_train_model,
                                    model_dir=train_dir,
                                    config=run_config,
                                    params={})

    tf.logging.info('start training model')
    model.train(input_fn=lambda :input_pipeline(num_epochs=45), hooks=None, max_steps=25000)
    tf.logging.info('Finish training model')

    finetune_model = tf.estimator.Estimator(model_fn=bcnn_finetune_model, 
                                            model_dir=cfg.finetune_dir,
                                            config=run_config,
                                            params={})
    tf.logging.info('start finetuning model')
    finetune_model.train(input_fn=lambda :input_pipeline(num_epochs=20), hooks=None)
    tf.logging.info('finish finetuning model.')


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
