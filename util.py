import os
import numpy as np
from PIL import Image
from StringIO import StringIO

import words
from words import VOCABULARY_SIZE


IMG_SHAPE = (224,224)
MAX_WORDS = 3


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def expand(x):
    return np.expand_dims(x, axis=0)


def decode_jpg(jpg, preprocess=True):
    if jpg.startswith('\xFF\xD8'):
        # jpg is a JPG buffer
        img = Image.open(StringIO(jpg))
    else:
        # jpg is a filename
        img = Image.open(jpg)
    img = img.convert('RGB')
    if preprocess:
        img = img.resize(IMG_SHAPE)
    pixels = np.array(img).astype(float)
    if preprocess:
        pixels = imagenet_process(pixels)
    return pixels


def encode_jpg(pixels):
    img = Image.fromarray(pixels.astype(np.uint8)).convert('RGB')
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


def imagenet_process(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    return x[:, :, ::-1]


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


def show(jpg, box=None):
    pixels = decode_jpg(jpg, preprocess=False)
    if box:
        draw_box(pixels, box)
    with open('/tmp/example.jpg', 'w') as fp:
        fp.write(encode_jpg(pixels))
        os.system('imgcat /tmp/example.jpg')
