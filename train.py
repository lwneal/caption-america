import time
import sys
import os
from keras import models, layers
import resnet50
import numpy as np
from PIL import Image
from StringIO import StringIO

import words
from words import VOCABULARY_SIZE
import dataset_grefexp
from util import onehot, expand, decode_jpg, left_pad, predict, MAX_WORDS

GRU_SIZE = 1024
WORDVEC_SIZE = 300


def build_model():
    resnet = resnet50.ResNet50(include_top=True)
    for layer in resnet.layers:
        layer.trainable = False

    image_model = models.Sequential()
    image_model.add(resnet)
    image_model.add(layers.Dense(WORDVEC_SIZE))
    image_model.add(layers.RepeatVector(MAX_WORDS))

    language_model = models.Sequential()
    language_model.add(layers.Embedding(VOCABULARY_SIZE, WORDVEC_SIZE, input_length=MAX_WORDS, mask_zero=True))
    language_model.add(layers.GRU(GRU_SIZE, return_sequences=True))
    language_model.add(layers.TimeDistributed(layers.Dense(WORDVEC_SIZE)))
    
    model = models.Sequential()
    model.add(layers.Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(layers.GRU(GRU_SIZE, return_sequences=False))
    model.add(layers.Dense(VOCABULARY_SIZE))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], decay=.0001, lr=.001)
    return model


def example():
    jpg_data, box, text = dataset_grefexp.example()

    x_img = decode_jpg(jpg_data, box=box)
    indices = words.indices(text)

    idx = np.random.randint(0, len(indices))
    x_words = left_pad(indices[:idx][-MAX_WORDS:])
    y = onehot(indices[idx])
    return [x_img, x_words], y


def gen():
    while True:
        BATCH_SIZE = 32
        X_img = np.zeros((BATCH_SIZE,224,224,3))
        X_words = np.zeros((BATCH_SIZE, MAX_WORDS), dtype=int)
        Y = np.zeros((BATCH_SIZE, VOCABULARY_SIZE))
        for i in range(BATCH_SIZE):
            x, y = example()
            x_img, x_words = x
            X_img[i] = x_img
            X_words[i] = x_words
            Y[i] = y
        yield [X_img, X_words], Y


if __name__ == '__main__':
    model_filename = sys.argv[1]
    if os.path.exists(model_filename):
        model = models.load_model(model_filename)
    else:
        model = build_model()
    i = 0
    while True:
        model.fit_generator(gen(), samples_per_epoch=2**10, nb_epoch=4)
        model.save(model_filename)
        print("Results on cat/horse/dog after {}k examples:", i * 4)
        print(predict(model, decode_jpg('cat.jpg')))
        print(predict(model, decode_jpg('horse.jpg')))
        print(predict(model, decode_jpg('dog.jpg')))
