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
import util
from util import onehot, decode_jpg, left_pad, MAX_WORDS

GRU_SIZE = 1024
WORDVEC_SIZE = 300


def build_model():
    resnet = resnet50.ResNet50(include_top=True)
    for layer in resnet.layers:
        layer.trainable = False

    image_model = models.Sequential()
    image_model.add(resnet)
    image_model.add(layers.Dense(WORDVEC_SIZE, activation='tanh'))
    image_model.add(layers.RepeatVector(MAX_WORDS))

    language_model = models.Sequential()
    language_model.add(layers.Embedding(VOCABULARY_SIZE, WORDVEC_SIZE, input_length=MAX_WORDS, mask_zero=True))
    language_model.add(layers.GRU(GRU_SIZE, return_sequences=True))
    language_model.add(layers.TimeDistributed(layers.Dense(WORDVEC_SIZE, activation='tanh')))
    
    model = models.Sequential()
    model.add(layers.Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(layers.GRU(GRU_SIZE, return_sequences=False))
    model.add(layers.Dense(VOCABULARY_SIZE, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], decay=.0001, lr=.001)
    return model


def training_generator():
    while True:
        BATCH_SIZE = 32
        X_img = np.zeros((BATCH_SIZE,224,224,3))
        X_words = np.zeros((BATCH_SIZE, MAX_WORDS), dtype=int)
        Y = np.zeros((BATCH_SIZE, VOCABULARY_SIZE))
        for i in range(BATCH_SIZE):
            x, y = process(*dataset_grefexp.example())
            x_img, x_words = x
            X_img[i] = x_img
            X_words[i] = x_words
            Y[i] = y
        yield [X_img, X_words], Y


def validation_generator():
    for k in dataset_grefexp.get_all_keys():
        x, y = process(*dataset_grefexp.get_annotation_for_key(k))
        yield x, y


def process(jpg_data, box, text):
    x_img = util.decode_jpg(jpg_data, box=box)
    indices = words.indices(text)
    idx = np.random.randint(0, len(indices))
    x_words = util.left_pad(indices[:idx][-MAX_WORDS:])
    y = util.onehot(indices[idx])
    return [x_img, x_words], y


def predict(model, X):
    return util.predict(model, X[0])


def demo(model):
    for f in ['cat.jpg', 'dog.jpg', 'horse.jpg', 'car.jpg']:
        print(util.predict(model, decode_jpg(f)))
