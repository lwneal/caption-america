import re
import os
import numpy as np
from PIL import Image
from StringIO import StringIO

import words
from words import VOCABULARY_SIZE

from imutil import *


IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
MAX_WORDS = 10


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def expand(x):
    return np.expand_dims(x, axis=0)


def left_pad(indices):
    res = np.zeros(MAX_WORDS, dtype=int)
    res[MAX_WORDS - len(indices):] = indices
    return res


def strip(text):
    # Remove the START_TOKEN
    text = text.replace('000', '')
    # Remove all text after the first END_TOKEN
    end_idx = text.find('001')
    if end_idx >= 0:
        text = text[:end_idx]
    # Remove non-alphanumeric characters and lowercase everything
    return re.sub(r'\W+', ' ', text.lower()).strip()

