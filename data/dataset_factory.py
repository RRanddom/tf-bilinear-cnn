from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import tensorflow as tf
from data.make_tfrecord import _parse_function

_FILE_PATTERN = '%s-*.tfrecord'

def get_dataset_dir(dataset_name):
    
    if dataset_name == 'cub200':
        return '/data/CUB_200_2011/CUB_200_2011/tfrecord/'

    elif dataset_name == 'aircraft':
        return '/data/fgvc-aircraft-2013b/data/tfrecord'

    elif dataset_name == 'standfordcars':
        return '/data/standford-cars/tfrecord/'

    else:
        raise ValueError('dataset_name '+str(dataset_name) +' unknown')

        

def get_dataset(dataset_name, split_name):
    """
    Gets an instance of  tf.data.Dataset
    Args:
        dataset_name: Dataset name.
        split_name: A train/val/test split name.

    Returns:
        An instance of  tf.data.Dataset
    """

    dataset_dir = get_dataset_dir(dataset_name)
    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)
    filenames = tf.gfile.Glob(file_pattern)
    
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset.map(_parse_function)
