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


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def expand(x, batch_size=None):
    if batch_size is not None:
        return np.array([x] * batch_size)
    return np.expand_dims(x, axis=0)


def left_pad(indices, **params):
    max_words = params['max_words']
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res


def right_pad(indices, **params):
    max_words = params['max_words']
    res = np.zeros(max_words, dtype=int)
    res[:len(indices)] = indices
    return res


def strip(text, strip_end=True):
    # Remove the START_TOKEN
    text = text.replace('000', '')
    if strip_end:
        # Remove all text after the first END_TOKEN
        end_idx = text.find('001')
        if end_idx >= 0:
            text = text[:end_idx]
    # Remove non-alphanumeric characters and lowercase everything
    return re.sub(r'\W+', ' ', text.lower()).strip()

