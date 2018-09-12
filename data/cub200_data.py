from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from PIL import Image
import numpy as np
import os
import random

import tensorflow as tf 
from data.make_tfrecord import ImageReader, image_to_tfexample
import math
import sys

birds_classes = []

cub200_classes = ["Black_footed_Albatross", "Laysan_Albatross", "Sooty_Albatross", "Groove_billed_Ani", "Crested_Auklet", "Least_Auklet", \
"Parakeet_Auklet", "Rhinoceros_Auklet", "Brewer_Blackbird", "Red_winged_Blackbird", "Rusty_Blackbird", "Yellow_headed_Blackbird", "Bobolink", \
"Indigo_Bunting", "Lazuli_Bunting", "Painted_Bunting", "Cardinal", "Spotted_Catbird", "Gray_Catbird", "Yellow_breasted_Chat", "Eastern_Towhee",\
"Chuck_will_Widow", "Brandt_Cormorant", "Red_faced_Cormorant", "Pelagic_Cormorant", "Bronzed_Cowbird", "Shiny_Cowbird", "Brown_Creeper",\
"American_Crow", "Fish_Crow", "Black_billed_Cuckoo", "Mangrove_Cuckoo", "Yellow_billed_Cuckoo", "Gray_crowned_Rosy_Finch", "Purple_Finch", \
"Northern_Flicker", "Acadian_Flycatcher", "Great_Crested_Flycatcher", "Least_Flycatcher", "Olive_sided_Flycatcher", "Scissor_tailed_Flycatcher", \
"Vermilion_Flycatcher", "Yellow_bellied_Flycatcher", "Frigatebird", "Northern_Fulmar", "Gadwall", "American_Goldfinch", "European_Goldfinch", \
"Boat_tailed_Grackle", "Eared_Grebe", "Horned_Grebe", "Pied_billed_Grebe", "Western_Grebe", "Blue_Grosbeak", "Evening_Grosbeak", "Pine_Grosbeak", \
"Rose_breasted_Grosbeak", "Pigeon_Guillemot", "California_Gull", "Glaucous_winged_Gull", "Heermann_Gull", "Herring_Gull", "Ivory_Gull", \
"Ring_billed_Gull", "Slaty_backed_Gull", "Western_Gull", "Anna_Hummingbird", "Ruby_throated_Hummingbird", "Rufous_Hummingbird", "Green_Violetear", \
"Long_tailed_Jaeger", "Pomarine_Jaeger", "Blue_Jay", "Florida_Jay", "Green_Jay", "Dark_eyed_Junco", "Tropical_Kingbird", "Gray_Kingbird", \
"Belted_Kingfisher", "Green_Kingfisher", "Pied_Kingfisher", "Ringed_Kingfisher", "White_breasted_Kingfisher", "Red_legged_Kittiwake", \
"Horned_Lark", "Pacific_Loon", "Mallard", "Western_Meadowlark", "Hooded_Merganser", "Red_breasted_Merganser", "Mockingbird", "Nighthawk", \
"Clark_Nutcracker", "White_breasted_Nuthatch", "Baltimore_Oriole", "Hooded_Oriole", "Orchard_Oriole", "Scott_Oriole", "Ovenbird", "Brown_Pelican", \
"White_Pelican", "Western_Wood_Pewee", "Sayornis", "American_Pipit", "Whip_poor_Will", "Horned_Puffin", "Common_Raven", "White_necked_Raven", \
"American_Redstart", "Geococcyx", "Loggerhead_Shrike", "Great_Grey_Shrike", "Baird_Sparrow", "Black_throated_Sparrow", "Brewer_Sparrow", \
"Chipping_Sparrow", "Clay_colored_Sparrow", "House_Sparrow", "Field_Sparrow", "Fox_Sparrow", "Grasshopper_Sparrow", "Harris_Sparrow", \
"Henslow_Sparrow", "Le_Conte_Sparrow", "Lincoln_Sparrow", "Nelson_Sharp_tailed_Sparrow", "Savannah_Sparrow", "Seaside_Sparrow", "Song_Sparrow", \
"Tree_Sparrow", "Vesper_Sparrow", "White_crowned_Sparrow", "White_throated_Sparrow", "Cape_Glossy_Starling", "Bank_Swallow", "Barn_Swallow", \
"Cliff_Swallow", "Tree_Swallow", "Scarlet_Tanager", "Summer_Tanager", "Artic_Tern", "Black_Tern", "Caspian_Tern", "Common_Tern", "Elegant_Tern", \
"Forsters_Tern", "Least_Tern", "Green_tailed_Towhee", "Brown_Thrasher", "Sage_Thrasher", "Black_capped_Vireo", "Blue_headed_Vireo", \
"Philadelphia_Vireo", "Red_eyed_Vireo", "Warbling_Vireo", "White_eyed_Vireo", "Yellow_throated_Vireo", "Bay_breasted_Warbler", "Black_and_white_Warbler",\
 "Black_throated_Blue_Warbler", "Blue_winged_Warbler", "Canada_Warbler", "Cape_May_Warbler", "Cerulean_Warbler", "Chestnut_sided_Warbler", \
 "Golden_winged_Warbler", "Hooded_Warbler", "Kentucky_Warbler", "Magnolia_Warbler", "Mourning_Warbler", "Myrtle_Warbler", "Nashville_Warbler", \
 "Orange_crowned_Warbler", "Palm_Warbler", "Pine_Warbler", "Prairie_Warbler", "Prothonotary_Warbler", "Swainson_Warbler", "Tennessee_Warbler", \
 "Wilson_Warbler", "Worm_eating_Warbler", "Yellow_Warbler", "Northern_Waterthrush", "Louisiana_Waterthrush", "Bohemian_Waxwing", "Cedar_Waxwing", 
 "American_Three_toed_Woodpecker", "Pileated_Woodpecker", "Red_bellied_Woodpecker", "Red_cockaded_Woodpecker", "Red_headed_Woodpecker", \
 "Downy_Woodpecker", "Bewick_Wren", "Cactus_Wren", "Carolina_Wren", "House_Wren", "Marsh_Wren", "Rock_Wren", "Winter_Wren", "Common_Yellowthroat",]


cub_200_root = "/data/CUB_200_2011/CUB_200_2011"
img_dir = os.path.join(cub_200_root, 'images')


def convert_cub200_to_tfrecord():
    dest_dir = os.path.join(cub_200_root, 'tfrecord')

    if not tf.gfile.Exists(dest_dir):
        tf.gfile.MkDir(dest_dir)
    
    image_reader = ImageReader('jpeg')

    images_list_file = os.path.join(cub_200_root, 'images.txt')  #image_id file_name
    image_id_class_file = os.path.join(cub_200_root, 'image_class_labels.txt')  # image_id class_label
    image_train_test_split_file = os.path.join(cub_200_root, 'train_test_split.txt')# image_id belong_to_train


    all_images = tf.gfile.FastGFile(images_list_file).readlines()
    image_class_list = tf.gfile.FastGFile(image_id_class_file).readlines()
    image_train_list = tf.gfile.FastGFile(image_train_test_split_file).readlines()
    
    train_list = [item.split()[0] for item in image_train_list if int(item.split()[-1])==1] #
    test_list = [item.split()[0] for item in image_train_list if int(item.split()[-1])!=1] #

    imageid_to_file = {}
    imageid_to_class = {}

    for item in all_images:
        img_id, img_file = item.split()
        imageid_to_file.update({img_id : img_file})

    for item in image_class_list:
        img_id, img_class = item.split()
        imageid_to_class.update({img_id : img_class})


    splits = ['train', 'test']
    _NUM_SHARDS = 4
    

    split = splits[0]
    num_per_shard = int(math.ceil(len(train_list) / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(dest_dir, '%s-%02d-of-%02d.tfrecord' %(split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = 0
            end_idx = min((shard_id + 1) * num_per_shard, len(train_list))

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> converting image %d/%d shard %d' % (
                    i + 1, len(train_list), shard_id))
                sys.stdout.flush()

                imgid = train_list[i]
                image_name = imageid_to_file[imgid]
                full_path = os.path.join(img_dir, image_name)
                img_class = int(imageid_to_class[imgid]) - 1
                img_label_desc = cub200_classes[img_class]
                image_data = tf.gfile.FastGFile(full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(image_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, image_name, height, width, img_class, img_label_desc)
                    tfrecord_writer.write(image_record.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()


    split = splits[1]
    num_per_shard = int(math.ceil(len(test_list) / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(dest_dir, '%s-%02d-of-%02d.tfrecord' %(split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = 0
            end_idx = min((shard_id + 1) * num_per_shard, len(test_list))

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> converting image %d/%d shard %d' % (
                    i + 1, len(test_list), shard_id))
                sys.stdout.flush()

                imgid = test_list[i]
                image_name = imageid_to_file[imgid]
                full_path = os.path.join(img_dir, image_name)
                img_class = int(imageid_to_class[imgid]) - 1
                img_label_desc = cub200_classes[img_class]
                image_data = tf.gfile.FastGFile(full_path, 'rb').read()
                height, width, channel = image_reader.read_image_dims(image_data)

                if channel != 3:
                    print ("\nimage:{} channel number:{}, not a legal rgb image".format(image_name, channel))
                else:
                    image_record = image_to_tfexample(image_data, image_name, height, width, img_class, img_label_desc)
                    tfrecord_writer.write(image_record.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    convert_cub200_to_tfrecord()
