from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
import numpy as np
import os

from data.make_tfrecord import ImageReader, image_to_tfexample
import math
import sys

stanford_car_devkit = "/data/standford-cars/devkit"


root_dir = "/data/standford-cars/"

train_img_dir = os.path.join(root_dir, "cars_train")
test_img_dir = os.path.join(root_dir, "cars_test")

img_annot_for_train = os.path.join(root_dir, "devkit/cars_train_annos.pkl")
img_annot_for_test = os.path.join(root_dir, "devkit/cars_test_annos.pkl")


def convert_standfordcars_to_tfrecord():
    dest_dir = os.path.join(root_dir, 'tfrecord')

    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MkDir(dest_dir)

    image_reader = ImageReader('jpeg')

    images_list_file = os.path.join(root_dir, 'images.txt')  #image_id file_name
    image_id_class_file = os.path.join(root_dir, 'image_class_labels.txt')  # image_id class_label
    image_train_test_split_file = os.path.join(root_dir, 'train_test_split.txt')# image_id belong_to_train


    all_training_imgs = pickle.load(open(img_annot_for_train, 'r'))
    all_testing_imgs = pickle.load(open(img_annot_for_test, 'r'))


    splits = ['train', 'test']
    _NUM_SHARDS = 4

    split = splits[0]
    num_per_shard = int(math.ceil(len(all_training_imgs) / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(dest_dir, '%s-%02d-of-%02d.tfrecord' %(split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = 0
            end_idx = min((shard_id + 1) * num_per_shard, len(all_training_imgs))

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> converting image %d/%d shard %d' % (
                    i + 1, len(all_training_imgs), shard_id))
                sys.stdout.flush()

                img_item = all_training_imgs[i] #{'img_name': u'05461.jpg', 'cls': 101}
                img_name = img_item['img_name']
                img_full_path = os.path.join(train_img_dir, img_name)
                img_cls = img_item['cls']

                img_label_desc = ''
                image_data = tf.gfile.FastGFile(img_full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(image_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, img_name, height, width, img_cls, img_label_desc)
                    tfrecord_writer.write(image_record.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()


    split = splits[1]
    num_per_shard = int(math.ceil(len(all_testing_imgs) / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(dest_dir, '%s-%02d-of-%02d.tfrecord' %(split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = 0
            end_idx = min((shard_id + 1) * num_per_shard, len(all_testing_imgs))

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> converting image %d/%d shard %d' % (
                    i + 1, len(all_testing_imgs), shard_id))
                sys.stdout.flush()

                img_item = all_testing_imgs[i] #{'img_name': u'05461.jpg', 'cls': 101}
                img_name = img_item['img_name']
                img_full_path = os.path.join(test_img_dir, img_name)
                img_cls = img_item['cls']

                img_label_desc = ''
                image_data = tf.gfile.FastGFile(img_full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(image_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, img_name, height, width, img_cls, img_label_desc)
                    tfrecord_writer.write(image_record.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()

if __name__ == '__main__':
    convert_standfordcars_to_tfrecord()
