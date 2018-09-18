from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from model.vgg import vgg_conv_body
from model.config import cfg

def bilinear_cnn(inputs, is_training, fine_tuning, num_class=-1):

    assert num_class>0, 'num_class should > 0'

    cnn_conv5_3, _ = vgg_conv_body(inputs, is_training=(fine_tuning and is_training))
    conv5_3 = tf.identity(cnn_conv5_3, 'conv5_3')

    with tf.variable_scope("sum-pooling"):
        conv5_3 = tf.transpose(conv5_3, perm=[0, 3, 1 , 2]) # shift to [Batch, Channel, Height, Width] ==> [B,512,28,28]
        conv5_3 = tf.reshape(conv5_3, [-1,512,28*28])
        conv5_3_T = tf.transpose(conv5_3, perm=[0,2,1])

        x_value = tf.matmul(conv5_3, conv5_3_T)
        x_value = tf.reshape(x_value, [-1, 512*512])
        x_value /= (28*28)
        y_value = tf.sqrt(x_value + 1e-10)
        z_value = tf.nn.l2_normalize(y_value, axis=1)

    with tf.variable_scope("fc-layer"):
        z_value = slim.dropout(z_value, 0.5, is_training=is_training)
        fc_net = slim.fully_connected(z_value, num_class, biases_initializer=tf.constant_initializer(1.0), trainable=is_training, activation_fn=None, normalizer_fn=None)
    
    return fc_net


