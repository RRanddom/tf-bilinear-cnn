from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from PIL import Image, ImageOps
import numpy as np
import os
import random

class dataset():
    def __init__(self, rand_seed=42, annot_file="", img_dir="", name="", batch_size=32):
        
        assert os.path.exists(annot_file), "annot file does not exist"
        assert os.path.exists(img_dir), "image dir does not exist"

        self._iter = 0
        self.name = name
        self.img_dir = img_dir
        self.batch_size = batch_size

        self.img_and_annot = pickle.load(open(annot_file, 'r'))
        random.seed(rand_seed)
        random.shuffle(self.img_and_annot)
        self.total_ct = len(self.img_and_annot)

    def process_img(self, img_full_path):
        '''
        read the image and return ndarray.
        '''
        raise NotImplementedError

    def get_next_batch(self):
        items = []
        img_batch = []
        cls_batch = []

        if self.batch_size + self._iter > self.total_ct:
            items = self.img_and_annot[self._iter:] + self.img_and_annot[:self._iter+self.batch_size-self.total_ct]
        else:
            items = self.img_and_annot[self._iter:self._iter+self.batch_size]

        self._iter = (self._iter + self.batch_size) % self.total_ct            

        for item in items:
            img_name = item["img_name"]
            cls = item["cls"]
        
            imgnd = self.process_img(os.path.join(self.img_dir, img_name))

            img_batch.append(imgnd)
            cls_batch.append(cls)

        img_batch = np.array(img_batch)
        cls_batch = np.array(cls_batch)
        
        return {"img":img_batch, "cls":cls_batch}
    
    def data_iterator(self, num_epochs=25):
        iter_times =  int(self.total_ct / self.batch_size * num_epochs)
        print ("total iteration:"+str(iter_times))
        for _ in range(iter_times):
            yield self.get_next_batch()
