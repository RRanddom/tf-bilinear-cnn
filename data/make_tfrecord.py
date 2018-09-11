import collections
import six
import tensorflow as tf 
import os
import sys

class ImageReader(object):
    
    def __init__(self, image_format='jpeg', channels=3):
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._sess = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data, 
                                                    channels=channels)
            elif self._image_format in ('png'):
                self._decode = tf.image.decode_png(self._decode_data, 
                                                    channels=channels)
        
    def read_image_dims(self, image_data):
        """
        Decodes the Image data string.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]
    
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


def image_to_tfexample(image_data, img_name, height, width, class_label):
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
        'image/label': _int64_list_feature(class_label)
    }))

# imgs_annot_for_train = "/data/CUB_200_2011/CUB_200_2011/train.pkl"
# imgs_annot_for_test = "/data/CUB_200_2011/CUB_200_2011/test.pkl"
cub_img_dir = "/data/CUB_200_2011/CUB_200_2011/images"

# 001.Black_footed_Albatross

def convert_cub200_to_tfrecord(img_dir):
    dir_names = tf.gfile.ListDirectory(img_dir)
    output_dir = os.path.join('/data/CUB_200_2011/CUB_200_2011', 'tfrecord')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_reader = ImageReader('jpeg', channels=3)

    split = 'train'
    shard_id = 0
    _NUM_SHARDS = 1
    output_filename = os.path.join(
            output_dir, '%s-%05d-of-%05d.tfrecord' % (split, shard_id, _NUM_SHARDS))
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:    
        for dir_name in dir_names:
            class_label = int(dir_name.split('.')[0]) - 1
            #all_images
            images = tf.gfile.Glob(os.path.join(img_dir, dir_name) + '/*.jpg')
            sys.stdout.write("Processing label:"+str(class_label))

            for img_name in images:
                image_data = tf.gfile.FastGFile(img_name, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                img_record = image_to_tfexample(image_data, img_name, height, width, class_label)
                tfrecord_writer.write(img_record.SerializeToString())
        
            sys.stdout.write('\n')
    
    sys.stdout.flush()

if __name__ == '__main__':
    convert_cub200_to_tfrecord(cub_img_dir)




