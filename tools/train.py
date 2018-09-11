from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os 
import numpy as np
import tensorflow as tf

from data.dataset_factory import get_dataset
from tools.config import cfg
from model.bilinear_cnn import BilinearCnn,bilinear_cnn
from model.vgg import Vgg



def _parse_function(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/label': tf.FixedLenFeature(
            (), tf.int64)
        )
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    with tf.variable_scope('decoder'):
        input_image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        input_height = parsed_features['image/height']
        input_width = parsed_features['image/width']
        image_name = parsed_features['image/filename']
        image_label = parsed_features['image/label']

    return input_image, input_height, input_width, image_name, image_label


    # dataset = tf.data.TFRecordDataset(['/data/CUB_200_2011/CUB_200_2011/tfrecord/train-00000-of-00001.tfrecord'])
    # dataset.map(_parse_function)




def _preprocess_for_training(input_image, height, width, image_name, label):
    processed_image = tf.cast(input_image, tf.float32)

    if label_image is not None:
        label_image = tf.cast(label_image, tf.int32)

    processed_image = tf.image.resize_images(input_image, size=(488,488))

    return processed_image, height, width, image_name, label





def input_pipeline(num_epochs=25):
    tf_record_file = '/data/CUB_200_2011/CUB_200_2011/tfrecord/train-00000-of-00001.tfrecord'
    dataset = tf.data.TFRecordDataset([tf_record_file])
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(_preprocess_for_training)
    dataset = dataset.shuffle(buffer_size=500).repeat(num_epochs).batch(12)

    iterator = dataset.make_one_shot_iterator()
    input_image, height, width, image_name, label = iterator.get_next()
    return input_image, label




def get_init_fn_for_scaffold(vgg_pretrained_path, model_dir, exclude_vars=[]):

    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info("Ignore pretrained mobilenet path because a checkpoint file already exists")
        return None
    
    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
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
        saver.restore(session, mobilev2_path)
    
    return callback

def bcnn_stage1_model(features, labels, mode, params):
    """
    Args:
        features: input image batch
        labels:   image label
        mode: 'TRAIN' | 'EVAL' ...
        params: additional params
    """
    bcnn = BilinearCnn(is_training=True, fine_tuning=False, num_class=cfg.num_classes)
    loss = bcnn.loss
    optimizer = tf.train.MomentumOptimizer(learning_rate=.9, momentum=.9)
    train_op = optimizer.minimize(loss)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=None, loss=loss, train_op=train_op,
                                        scaffold=tf.train.Scaffold(init_fn=get_init_fn_for_scaffold(cfg.vgg_pretrained_path, cfg.train_dir_stage1)))



def bcnn_stage_finetune(features, labels, mode, params):

    # bcnn = BilinearCnn(is_training=True, fine_tuning=True, num_class=cfg.num_classes)
    # loss = bcnn.loss
    # def bilinear_cnn(images, is_training, fine_tuning, num_class=100):

    gt_labels = labels
    logits = bilinear_cnn(features, is_training=True, fine_tuning=False, num_class=cfg.num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(gt_labels, depth=cfg.num_class), logits=logits)

    optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=.9)
    train_op = optimizer.minimize(loss)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=None, loss=loss, train_op=train_op)



# def deeplab_seg_model(features, labels, mode, params):
#     input_image = features
#     label_image, height, width = labels

#     logits = deeplab_mobilev2(features, is_training=True, number_classes=FLAGS.num_classes)
    
#     loss = add_softmax_cross_entropy_loss(logits, label_image, num_classes=FLAGS.num_classes, ignore_label=FLAGS.ignore_label)
#     tf.summary.scalar('softmax_loss', loss)
#     global_step = tf.train.get_or_create_global_step()

#     learning_rate = get_model_learning_rate(
#         FLAGS.learning_policy, FLAGS.base_learning_rate,
#         FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
#         FLAGS.training_number_of_steps, FLAGS.learning_power,
#         FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)

#     tf.summary.scalar('learning_rate', learning_rate)
#     optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
#                                             momentum = FLAGS.momentum)

#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         train_op = optimizer.minimize(loss, global_step)

#     return tf.estimator.EstimatorSpec(
#                                 mode=mode,
#                                 predictions=logits,
#                                 loss=loss,
#                                 train_op=train_op,
#                                 scaffold=tf.train.Scaffold(
#                                     init_fn=get_init_fn_for_scaffold('/data/model/tf_model/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt', '/data/benchmark_RELEASE/dataset/model')))


def train_for_stage1(pretrained_vgg_path, log_dir, train_dir, learning_rate=0.9):
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




if __name__ == '__main__':
    '''
    train stage 1
    '''
    train_dir_stage1 = cfg.train_dir_stage1
    train_dir_stage2 = cfg.train_dir_stage2

    run_config = tf.estimator.RunConfig().replace(save_summary_steps=2500).replace(log_step_count_steps=10)
    
    model = tf.estimator.Estimator(model_fn=bcnn_stage1_model, model_dir=train_dir_stage1, config=run_config, params={})
    tf.logging.info("Start training model.")
    model.train(input_fn=input_pipeline, hooks=None)
    tf.logging.info("Finish training model.")





# def snapshot(sess, saver, _iter, train_dir):
    # if not os.path.exists(train_dir):
    #     os.makedirs(train_dir)
    
    # file_name = "bilinear"+"_"+str(_iter)+".ckpt"
    # full_path = os.path.join(train_dir, file_name)
    # saver.save(sess, full_path)
    # print ("write snapshot to:",full_path)


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

# if __name__ == "__main__":

#     vgg_path = cfg.vgg_pretrained_path

#     log_dir_stage1 = cfg.log_dir_stage1
#     log_dir_stage2 = cfg.log_dir_stage2

#     train_dir_stage1 = cfg.train_dir_stage1
#     train_dir_stage2 = cfg.train_dir_stage2

#     stage1_lr = .9
#     stage2_lr = 1e-3
    
#     restore_path = train_for_stage1(pretrained_vgg_path=vgg_path, log_dir=log_dir_stage1, train_dir=train_dir_stage1, learning_rate=stage1_lr)
#     tf.reset_default_graph()
#     train_for_stage2(ckpt_path=restore_path, log_dir=log_dir_stage2, train_dir=train_dir_stage2, learning_rate=stage2_lr)
