from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pickle
from PIL import Image, ImageOps
import numpy as np
import os
import random

from data.standford_cars import standfordcars_dataset
from data.cub200_data import cub200_dataset
from data.aircraft_data import aircraft_dataset


def get_dataset(name, mode="TRAIN"):
    if name=="cub200":
        return cub200_dataset(mode=mode)
    elif name=="aircraft":
        return aircraft_dataset(mode=mode)
    elif name=="standfordcars":
        return standfordcars_dataset(mode=mode)
    else:
        raise ValueError('Name of dataset unknown %s' % name)

