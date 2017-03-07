import os
import numpy as np
from PIL import Image
from StringIO import StringIO

import words
from words import VOCABULARY_SIZE


IMG_SHAPE = (224,224)
MAX_WORDS = 20


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def expand(x):
    return np.expand_dims(x, axis=0)


def decode_jpg(jpg):
    if jpg.startswith('\xFF\xD8'):
        # jpg is a JPG buffer
        img = Image.open(StringIO(jpg))
    else:
        # jpg is a filename
        img = Image.open(jpg)
    return np.array(img.convert('RGB').resize(IMG_SHAPE))


def left_pad(indices):
    res = np.zeros(MAX_WORDS, dtype=int)
    res[MAX_WORDS - len(indices):] = indices
    return res


def predict(model, img):
    indices = left_pad([])
    for _ in range(MAX_WORDS):
        preds = model.predict([expand(img), expand(indices)])
        indices = np.roll(indices, -1)
        indices[-1] = np.argmax(preds[0], axis=-1)
    return words.words(indices)
