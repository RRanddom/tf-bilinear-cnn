from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from PIL import Image
import numpy as np
import os
import random

import scipy.io as sio
import seaborn as sn
from matplotlib import pyplot as plt
from data.dataset import dataset

stanford_car_devkit = "/data/standford-cars/devkit"

train_img_dir = "/data/standford-cars/cars_train/"
test_img_dir = "/data/standford-cars/cars_test/"        

img_annot_for_train = "/data/standford-cars/devkit/cars_train_annos.pkl"
img_annot_for_test = "/data/standford-cars/devkit/cars_test_annos.pkl"

class standfordcars_dataset(dataset):
    def __init__(self, mode="TRAIN"):
        if mode == "TRAIN":
            dataset.__init__(self, annot_file=img_annot_for_train, img_dir=train_img_dir, name="standford_cars")
        else:
            dataset.__init__(self, annot_file=img_annot_for_test, img_dir=test_img_dir, name="standford_cars")
        
    def process_img(self, img_full_path):
        imgobj = Image.open(img_full_path)
        img_resized = imgobj.resize((448,448), Image.ANTIALIAS)
        imgnd = np.array(img_resized)
        if len(imgnd.shape) == 2: # grayscale image
            imgnd = np.stack((imgnd,) * 3, -1)
        elif imgnd.shape[2] == 4:
            imgnd = imgnd[...,:3] # rgba
        
        return imgnd
