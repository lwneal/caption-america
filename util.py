import re
import os
import numpy as np
from PIL import Image
from StringIO import StringIO

import words
from words import VOCABULARY_SIZE


IMG_SHAPE = (224,224)
MAX_WORDS = 10


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def expand(x):
    return np.expand_dims(x, axis=0)


# Swiss army knife for image decoding
def decode_jpg(jpg, box=None, crop_to_box=None, preprocess=True):
    if jpg.startswith('\xFF\xD8'):
        # jpg is a JPG buffer
        img = Image.open(StringIO(jpg))
    else:
        # jpg is a filename
        img = Image.open(jpg)
    img = img.convert('RGB')
    width = img.width
    height = img.height
    if crop_to_box:
        # Crop to bounding box
        x0, x1, y0, y1 = crop_to_box
        img = img.crop((x0,y0,x1,y1))
    if preprocess:
        img = img.resize(IMG_SHAPE)
    pixels = np.array(img).astype(float)
    if preprocess:
        pixels = imagenet_process(pixels)
    if box:
        # Transform a bounding box after resizing
        x0, x1, y0, y1 = box
        xs = float(pixels.shape[1]) / width
        ys = float(pixels.shape[0]) / height
        x0 *= xs
        x1 *= xs
        y0 *= ys
        y1 *= ys
        return pixels, (x0, x1, y0, y1)
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



def show(jpg, box=None):
    if type(jpg) == type(np.array([])):
        pixels = jpg
    else:
        pixels = decode_jpg(jpg, preprocess=False)
    if box:
        draw_box(pixels, box)
    with open('/tmp/example.jpg', 'w') as fp:
        fp.write(encode_jpg(pixels))
        os.system('imgcat /tmp/example.jpg')


def draw_box(img, box, color=1.0):
    x0, x1, y0, y1 = (int(val) for val in box)
    height, width, channels = img.shape
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    img[y0:y1,x0] = color
    img[y0:y1,x1] = color
    img[y0,x0:x1] = color
    img[y1,x0:x1] = color


def strip(text):
    # Remove the START_TOKEN
    text = text.replace('000', '')
    # Remove all text after the first END_TOKEN
    end_idx = text.find('001')
    if end_idx >= 0:
        text = text[:end_idx]
    # Remove non-alphanumeric characters and lowercase everything
    return re.sub(r'\W+', ' ', text.lower()).strip()
