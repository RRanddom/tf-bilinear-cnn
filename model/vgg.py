from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import init_ops


class Vgg(object):

    def __init__(self, is_training=True):
        self.is_training = is_training

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.
        Args:
            weight_decay: The l2 regularization coefficient.
        Returns:
            An arg_scope.
        """
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            biases_initializer=init_ops.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def convbody(self, image_input):
        is_training=self.is_training
        with slim.arg_scope(self.vgg_arg_scope()):
            with tf.variable_scope("vgg_16"):
                collections = {}
                net = slim.repeat(image_input, 2, slim.conv2d, 64, [3,3], trainable=is_training, scope='conv1')
                collections.update({"conv1" : net})
                net = slim.max_pool2d(net, [2,2], scope='pool1')
                collections.update({"pool1" : net})

                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=is_training, scope='conv2')
                collections.update({"conv2" : net})
                net = slim.max_pool2d(net, [2, 2,], scope='pool2')
                collections.update({"pool2" : net})

                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
                collections.update({"conv3" : net})
                net = slim.max_pool2d(net, [2, 2],scope='pool3')
                collections.update({"pool3" : net})

                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')
                collections.update({"conv4" : net})
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                collections.update({"pool4" : net})

                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv5')
                collections.update({"conv5" : net})
                            
        return net, collections
        