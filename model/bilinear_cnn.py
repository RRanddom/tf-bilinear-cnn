from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from model.vgg import vgg_conv_body
from tools.config import cfg


# _R_MEAN, _G_MEAN, _B_MEAN = cfg._RGB_MEAN


# vgg = Vgg(is_training=False)

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

# rgb_mean = tf.reshape(np.array([_R_MEAN, _G_MEAN, _B_MEAN]).astype(np.float32), [1,1,1,3])
# img = images - rgb_mean

# if is_training:

#     resized = tf.image.resize_images(img, (488, 488))

#     crop_fn = lambda x: tf.random_crop(x, [448, 448, 3])
#     processed = tf.map_fn(crop_fn, resized)

#     flip_fn = lambda x: tf.image.random_flip_left_right(x)
#     processed = tf.map_fn(flip_fn, processed)

#     # brightness_fn = lambda x: tf.image.random_brightness(x, max_delta=0.2)
#     # processed = tf.map_fn(brightness_fn, processed)

# else:
#     processed = tf.image.resize_images(img, (448, 448))

# img_after_preprocess = tf.identity(processed)

##########################################################################################################

# class BilinearCnn(object):
    
#     def __init__(self, is_training, fine_tuning, num_class=100):
        
#         self.vgg_network_trainable = (fine_tuning and is_training)

#         self.is_training = is_training

#         self.num_class = num_class
        
#         self.image_input = tf.placeholder(tf.float32, shape=[None, 448, 448, 3])
#         if self.is_training:
#             self.gt_label = tf.placeholder(tf.int32, shape=[None])
        
#         self.logits = self._main_architecture()

#         print ("logits shape:{}".format(self.logits))

#         predictions = tf.argmax(input=self.logits, axis=1)

#         self.predictions = predictions

#         if self.is_training:
#             self.loss   = self._add_losses()
#             tf.summary.scalar("cls_loss", self.loss)

#         self.summary_op = tf.summary.merge_all()


#     def preprocessing(self, img):        

#         rgb_mean = tf.reshape(np.array([_R_MEAN, _G_MEAN, _B_MEAN]).astype(np.float32), [1,1,1,3])
#         img = img - rgb_mean
        
#         if self.is_training:

#             resized = tf.image.resize_images(img, (488, 488))

#             crop_fn = lambda x: tf.random_crop(x, [448, 448, 3])
#             processed = tf.map_fn(crop_fn, resized)

#             flip_fn = lambda x: tf.image.random_flip_left_right(x)
#             processed = tf.map_fn(flip_fn, processed)

#             # brightness_fn = lambda x: tf.image.random_brightness(x, max_delta=0.2)
#             # processed = tf.map_fn(brightness_fn, processed)

#         else:
#             processed = tf.image.resize_images(img, (448, 448))
                
#         return processed

#     def _main_architecture(self):

#         img_after_preprocess = self.preprocessing(self.image_input)
        
#         vgg = Vgg(is_training=self.vgg_network_trainable)
        
#         cnn_conv5_3, _ = vgg.convbody(img_after_preprocess)
#         self.conv5_3 = cnn_conv5_3
        
#         with tf.variable_scope("sum-pooling"):
#             cnn_conv5_3 = tf.transpose(cnn_conv5_3, perm=[0, 3, 1 , 2]) # shift to [Batch, Channel, Height, Width] ==> [B,512,28,28]
#             cnn_conv5_3 = tf.reshape(cnn_conv5_3, [-1,512,28*28])
#             cnn_conv5_3_T = tf.transpose(cnn_conv5_3, perm=[0,2,1])

#             x_value = tf.matmul(cnn_conv5_3, cnn_conv5_3_T)
#             x_value = tf.reshape(x_value, [-1, 512*512])
#             x_value /= (28*28)
#             y_value = tf.sqrt(x_value + 1e-10)
#             z_value = tf.nn.l2_normalize(y_value, axis=1)
#             self.z_value = z_value

#         with tf.variable_scope("fc-layer"):
#             z_value = slim.dropout(z_value, 0.5, is_training=self.is_training)
#             fc_net = slim.fully_connected(z_value, self.num_class, biases_initializer=tf.constant_initializer(1.0), trainable=self.is_training, activation_fn=None, normalizer_fn=None)
    
#         return fc_net


#     def train_step(self, sess, blob, train_op):
#         return sess.run([self.loss, train_op], feed_dict={self.image_input:blob["img"], self.gt_label:blob["cls"]})

#     def train_step_with_summary(self, sess, blob, train_op):
#         return sess.run([self.loss, self.summary_op, train_op], feed_dict={self.image_input:blob["img"], self.gt_label:blob["cls"]})

#     def test_step(self, sess, blob):
#         return sess.run(self.predictions, feed_dict={self.image_input:blob["img"]})
            
#     def detail_info(self, sess, imgnd):
#         if len(imgnd.shape) == 3:
#             imgnd = np.expand_dims(imgnd, 0)

#         pred, conv5_3, z_value = sess.run([self.predictions, self.conv5_3, self.z_value], feed_dict={self.image_input:imgnd})
#         return pred, conv5_3, z_value

#     def _add_losses(self):
#         cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.gt_label, depth=self.num_class), logits=self.logits)
#         return cls_loss
        

# if __name__ == "__main__":
#     bcnn = BilinearCnn(is_training=True, fine_tuning=False)
    
