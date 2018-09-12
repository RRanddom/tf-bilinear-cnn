import collections
import six
import tensorflow as tf 
import os
import sys

class ImageReader(object):
    
    def __init__(self, image_format='jpeg'):
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._sess = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data)
            elif self._image_format in ('png'):
                self._decode = tf.image.decode_png(self._decode_data)
        
    def read_image_dims(self, image_data):
        """
        Decodes the Image data string.
        """
        image = self.decode_image(image_data)
        return image.shape[:3]
    
    def decode_image(self, image_data):
        """
        Decodes the image data string.
        
        Args:
            image_data : string or image data
        """
        image = self._sess.run(self._decode,
                            feed_dict={self._decode_data : image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1,3):
            raise ValueError('The image channels not supported.')
        
        return image

def  _int64_list_feature(values):
    """
    Returns a TF-Feature of int64_list

    Args:
        values: A scaler or list of values

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_list_feature(values):
    """
    Returns a TF-Feature or bytes
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value
    
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))

def image_to_tfexample(image_data, img_name, height, width, class_label, class_desc):
    """
    Converts one image/segmentation pair to tf example

    Args:
        image_data:
        filename:
        height:
        width
        seg_data: string of semantic segmentation data.    
    """

    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_list_feature(image_data),
        'image/filename': _bytes_list_feature(img_name),
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/label': _int64_list_feature(class_label),
        'image/labeldesc': _bytes_list_feature(class_desc)
    }))


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
            (), tf.int64),
        'image/labeldesc': tf.FixedLenFeature(
            (), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    with tf.variable_scope('decoder'):
        input_image = tf.image.decode_jpeg(parsed_features['image/encoded'])
        input_height = parsed_features['image/height']
        input_width = parsed_features['image/width']
        image_name = parsed_features['image/filename']
        image_label = parsed_features['image/label']
        label_desc = parsed_features['image/labeldesc']

    return input_image, input_height, input_width, image_name, image_label, label_desc


# def _preprocess_for_training(input_image, height, width, image_name, label):
#     processed_image = tf.cast(input_image, tf.float32)

#     processed_image = tf.image.resize_images(input_image, size=(488,488))
    
#     last_dim = tf.shape(processed_image)[-1]

#     return processed_image, height, width, image_name, label



# def input_pipeline(num_epochs=25):
#     tf_record_file = '/data/CUB_200_2011/CUB_200_2011/tfrecord/train-00000-of-00001.tfrecord'
#     dataset = tf.data.TFRecordDataset([tf_record_file])
#     dataset = dataset.map(_parse_function)
#     dataset = dataset.map(_preprocess_for_training)
#     dataset = dataset.shuffle(buffer_size=500).repeat(num_epochs).batch(12)

#     iterator = dataset.make_one_shot_iterator()
#     input_image, height, width, image_name, label = iterator.get_next()
#     return input_image, label


