from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from PIL import Image
import numpy as np
import os
import random
from data.dataset import dataset


aircraft_classes = ["707-320", "727-200", "737-200", "737-300", "737-400", "737-500", "737-600", "737-700", "737-800", "737-900", "747-100", "747-200", "747-300", "747-400", "757-200", "757-300", "767-200", "767-300", "767-400", "777-200", "777-300", "A300B4", "A310", "A318", "A319", "A320", "A321", "A330-200", "A330-300", "A340-200", "A340-300", "A340-500", "A340-600", "A380", "ATR-42", "ATR-72", "An-12", "BAE 146-200", "BAE 146-300", "BAE-125", "Beechcraft 1900", "Boeing 717", "C-130", "C-47", "CRJ-200", "CRJ-700", "CRJ-900", "Cessna 172", "Cessna 208", "Cessna 525", "Cessna 560", "Challenger 600", "DC-10", "DC-3", "DC-6", "DC-8", "DC-9-30", "DH-82", "DHC-1", "DHC-6", "DHC-8-100", "DHC-8-300", "DR-400", "Dornier 328", "E-170", "E-190", "E-195", "EMB-120", "ERJ 135", "ERJ 145", "Embraer Legacy 600", "Eurofighter Typhoon", "F-16A/B", "F/A-18", "Falcon 2000", "Falcon 900", "Fokker 100", "Fokker 50", "Fokker 70", "Global Express", "Gulfstream IV", "Gulfstream V", "Hawk T1", "Il-76", "L-1011", "MD-11", "MD-80", "MD-87", "MD-90", "Metroliner", "Model B200", "PA-28", "SR-20", "Saab 2000", "Saab 340", "Spitfire", "Tornado", "Tu-134", "Tu-154", "Yak-42"]

all_models_file = "/data/fgvc-aircraft-2013b/data/variants.txt"
imgs_annot_for_train = "/data/fgvc-aircraft-2013b/data/images_variant_trainval.pkl"
imgs_annot_for_test  = "/data/fgvc-aircraft-2013b/data/images_variant_test.pkl"
img_dir = "/data/fgvc-aircraft-2013b/data/images"


class aircraft_dataset(dataset):
    def __init__(self, mode="TRAIN"):
        if mode == "TRAIN":
            dataset.__init__(self, annot_file=imgs_annot_for_train, img_dir=img_dir, name="aircraft")
        else:
            dataset.__init__(self, annot_file=imgs_annot_for_test, img_dir=img_dir, name="aircraft")

    def process_img(self, img_full_path):
        imgobj = Image.open(img_full_path)
        width, height = imgobj.size 
        img_crop = imgobj.crop((0,0,width,height-20))
        img_resized = img_crop.resize((448,448), Image.ANTIALIAS)
        imgnd = np.array(img_resized)
        return imgnd