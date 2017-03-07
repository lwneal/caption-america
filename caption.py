import numpy as np
import sys
import os
from keras import models
from PIL import Image

from util import left_pad, predict, decode_jpg


if __name__ == '__main__':
    img_filename = sys.argv[1]
    model_filename = sys.argv[2]

    pixels = decode_jpg(img_filename)
    model = models.load_model(model_filename)
    print("Loaded Keras model {}".format(model))

    print(predict(model, pixels))
