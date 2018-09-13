from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle

import numpy as np
import os

from data.make_tfrecord import ImageReader, image_to_tfexample
import math
import sys
import tensorflow as tf

aircraft_classes = ["707-320", "727-200", "737-200", "737-300", "737-400", "737-500", "737-600", "737-700", "737-800", \
"737-900", "747-100", "747-200", "747-300", "747-400", "757-200", "757-300", "767-200", "767-300", "767-400", "777-200", \
"777-300", "A300B4", "A310", "A318", "A319", "A320", "A321", "A330-200", "A330-300", "A340-200", "A340-300", "A340-500", \
"A340-600", "A380", "ATR-42", "ATR-72", "An-12", "BAE 146-200", "BAE 146-300", "BAE-125", "Beechcraft 1900", "Boeing 717", \
"C-130", "C-47", "CRJ-200", "CRJ-700", "CRJ-900", "Cessna 172", "Cessna 208", "Cessna 525", "Cessna 560", "Challenger 600", \
"DC-10", "DC-3", "DC-6", "DC-8", "DC-9-30", "DH-82", "DHC-1", "DHC-6", "DHC-8-100", "DHC-8-300", "DR-400", "Dornier 328", "E-170", \
"E-190", "E-195", "EMB-120", "ERJ 135", "ERJ 145", "Embraer Legacy 600", "Eurofighter Typhoon", "F-16A/B", "F/A-18", "Falcon 2000", \
"Falcon 900", "Fokker 100", "Fokker 50", "Fokker 70", "Global Express", "Gulfstream IV", "Gulfstream V", "Hawk T1", "Il-76", "L-1011", \
"MD-11", "MD-80", "MD-87", "MD-90", "Metroliner", "Model B200", "PA-28", "SR-20", "Saab 2000", "Saab 340", "Spitfire", "Tornado", \
"Tu-134", "Tu-154", "Yak-42"]

aircraft_root = "/data/fgvc-aircraft-2013b/data/"
aircraft_img_dir = os.path.join(aircraft_root, "images")
train_annot_file = os.path.join(aircraft_root, "images_variant_trainval.pkl")
test_annot_file = os.path.join(aircraft_root, "images_variant_test.pkl")


def convert_aircraftdata_to_tfrecord():
    dest_dir = os.path.join(aircraft_root, 'tfrecord')

    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MkDir(dest_dir)
    
    image_reader = ImageReader('jpeg')

    all_training_imgs = pickle.load(open(train_annot_file, 'r'))
    all_testing_imgs = pickle.load(open(test_annot_file, 'r'))

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
                img_full_path = os.path.join(aircraft_img_dir, img_name)
                img_cls = img_item['cls']
                img_label_desc = aircraft_classes[img_cls]
                image_data = tf.gfile.FastGFile(img_full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(img_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, str(img_name), height, width, img_cls, img_label_desc)
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

                img_item = all_testing_imgs[i]
                img_name = img_item['img_name']
                img_full_path = os.path.join(aircraft_img_dir, img_name)
                img_cls = img_item['cls']

                img_label_desc = ''
                image_data = tf.gfile.FastGFile(img_full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(img_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, str(img_name), height, width, img_cls, img_label_desc)
                    tfrecord_writer.write(image_record.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()

if __name__ == '__main__':
    convert_aircraftdata_to_tfrecord()

