import numpy as np
import os
import random
import re
import sys
import time
from keras import models, layers
from PIL import Image
from StringIO import StringIO

import resnet50
import tensorflow as tf
import words
import dataset_grefexp
import bleu_scorer
import rouge_scorer
import util
from util import MAX_WORDS


def build_model(GRU_SIZE=1024, WORDVEC_SIZE=200, ACTIVATION='relu'):
    resnet = resnet50.ResNet50(include_top=True)
    for layer in resnet.layers[-1]:
        layer.trainable = False

    image_model = models.Sequential()
    image_model.add(resnet)
    image_model.add(layers.Dense(WORDVEC_SIZE, activation=ACTIVATION))
    image_model.add(layers.RepeatVector(MAX_WORDS))

    language_model = models.Sequential()
    language_model.add(layers.Embedding(words.VOCABULARY_SIZE, WORDVEC_SIZE, input_length=MAX_WORDS, mask_zero=True))
    language_model.add(layers.GRU(GRU_SIZE, return_sequences=True))
    language_model.add(layers.TimeDistributed(layers.Dense(WORDVEC_SIZE, activation=ACTIVATION)))

    model = models.Sequential()
    model.add(layers.Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(layers.GRU(GRU_SIZE, return_sequences=False))
    model.add(layers.Dense(words.VOCABULARY_SIZE, activation='softmax'))
    return model


# TODO: Move batching out to the generic runner
def training_generator():
    while True:
        BATCH_SIZE = 32
        X_img = np.zeros((BATCH_SIZE,224,224,3))
        X_words = np.zeros((BATCH_SIZE, MAX_WORDS), dtype=int)
        Coords = np.zeros((BATCH_SIZE, 3), dtype=int)
        Y = np.zeros((BATCH_SIZE, words.VOCABULARY_SIZE))
        for i in range(BATCH_SIZE):
            x, y = process(*dataset_grefexp.example())
            x_img, coords, x_words = x
            X_img[i] = x_img
            Coords[i] = (i,) + coords
            X_words[i] = x_words
            Y[i] = y
        yield [X_img, Coords, X_words], Y


def process(jpg_data, box, texts):
    x_img, box = util.decode_jpg(jpg_data, box)
    text = util.strip(random.choice(texts))
    indices = words.indices(text)
    idx = np.random.randint(0, len(indices))
    x_words = util.left_pad(indices[:idx][-MAX_WORDS:])
    y = util.onehot(indices[idx])
    coords = coords_from_box(box)
    return [x_img, coords, x_words], y


def coords_from_box(box):
    x0, x1, y0, y1 = box
    dy = (y0 + y1) / 2
    dx = (x0 + x1) / 2
    return dy / 32, dx / 32


def validation_generator():
    for k in dataset_grefexp.get_all_keys():
        jpg_data, box, texts = dataset_grefexp.get_annotation_for_key(k)
        x, y = process(jpg_data, box, texts)
        x_img, coords, x_words = x
        yield x_img, box, texts


def evaluate(model, x_img, box, texts):
    candidate = util.strip(util.predict(model, x_img, box))
    references = map(util.strip, texts)
    print("[1F[K{} ({})".format(candidate, references[0]))
    scores = {}
    scores['bleu1'], scores['bleu2'] = bleu(candidate, references)
    scores['rouge'] = rouge(candidate, references)
    return scores


def bleu(candidate, references):
    scores, _ = bleu_scorer.BleuScorer(candidate, references, n=2).compute_score(option='closest')
    return scores


def rouge(candidate, references):
    return rouge_scorer.Rouge().calc_score([candidate], references)


def demo(model):
    for f in ['cat.jpg', 'dog.jpg', 'horse.jpg', 'car.jpg']:
        img = util.decode_jpg(f)
        box = (0, img.shape[1], 0, img.shape[0])
        print("Prediction for {} {}:".format(f, box)),
        print(util.predict(model, img, box))
