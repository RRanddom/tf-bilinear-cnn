from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os 
import pickle

from model.bilinear_cnn import bilinear_cnn
from data.dataset_factory import get_dataset
from tools.config import cfg

def _preprocess_for_testing(input_image, input_height, input_width, image_name, image_label, label_desc):
    
    resized = tf.image.resize_images(input_image, (448, 448))

    return resized, input_height, input_width, image_name, image_label, label_desc

def input_pipeline():

    dataset = get_dataset(dataset_name=cfg.current_dataset, split_name='test')
    dataset = dataset.map(_preprocess_for_testing)
    dataset = dataset.repeat(1).batch(cfg.BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    input_image, input_height, input_width, image_name, image_label, label_desc = iterator.get_next()
    return input_image, image_label



def bcnn_infer(features, labels, mode, params):
    '''
        features: input image batch
        labels:   image label
        mode: 'TRAIN' | 'EVAL' | 'PREDICT'
        params: additional params
    '''
    logits = bilinear_cnn(features, is_training=False, fine_tuning=False, num_class=cfg.num_classes)
    predictions = tf.argmax(logits, axis=0)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=cfg.num_classes), logits=logits)
    tf.summary.scalar('softmax_loss', loss)
    global_step = tf.train.get_or_create_global_step()

    accuracy, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    tf.summary.scalar('accuracy', accuracy)

    return tf.estimator.EstimatorSpec(mode=mode, 
                                      predictions=logits,
                                      eval_metric_ops={'acc': (accuracy, update_op)})

def main(unused_argv):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    run_config = tf.estimator.RunConfig()\
                    .replace(save_summary_steps=2500)\
                    .replace(log_step_count_steps=10)

    train_dir = cfg.train_dir
    model = tf.estimator.Estimator(model_fn=bcnn_infer,
                                   model_dir=train_dir,
                                   config=run_config,
                                   params={})

    tf.logging.info('Start eval model')
    result = model.evaluate(input_fn=input_pipeline)
    tf.logging.info('Finish eval model')



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()