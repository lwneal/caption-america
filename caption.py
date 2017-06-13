import numpy as np
import os
import random
import re
import sys
import time
from keras import models, layers
from PIL import Image
from StringIO import StringIO

from keras.applications import resnet50
import tensorflow as tf
import words
import dataset_grefexp
import bleu_scorer
import rouge_scorer
import util
from util import IMG_HEIGHT, IMG_WIDTH, IMG_SHAPE, IMG_CHANNELS
from keras import applications

from cgru import SpatialCGRU

BATCH_SIZE = 16

def build_model(**params):
    # TODO: get all these from **params
    CNN = 'resnet'
    INCLUDE_TOP = False
    LEARNABLE_CNN_LAYERS = params['learnable_cnn_layers']
    RNN_TYPE = 'LSTM'
    RNN_SIZE = 1024
    WORDVEC_SIZE = params['wordvec_size']
    ACTIVATION = 'relu'
    USE_CGRU = params['use_cgru']
    CGRU_SIZE = params['cgru_size']
    REDUCE_MEAN = params['reduce_visual']
    max_words = params['max_words']

    if CNN == 'vgg16':
        cnn = applications.vgg16.VGG16(include_top=INCLUDE_TOP)
    elif CNN == 'resnet':
        cnn = applications.resnet50.ResNet50(include_top=INCLUDE_TOP)
        # Pop the mean pooling layer
        cnn = models.Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)

    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Context Vector input
    # normalized to [0,1] the values:
    # left, top, right, bottom, (box area / image area)
    input_ctx = layers.Input(batch_shape=(BATCH_SIZE, 5))
    ctx = layers.BatchNormalization()(input_ctx)
    repeat_ctx = layers.RepeatVector(max_words)(ctx)


    # Global Image featuers (convnet output for the whole image)
    input_img_global = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    image_global = cnn(input_img_global)

    # Add a residual CGRU layer
    if USE_CGRU:
        image_global = layers.Conv2D(CGRU_SIZE, (1,1), padding='same', activation='relu')(image_global)
        res_cgru = SpatialCGRU(image_global, CGRU_SIZE)
        image_global = layers.add([image_global, res_cgru])

    if REDUCE_MEAN:
        image_global = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(image_global)
        image_global = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(image_global)
    else:
        image_global = layers.Conv2D(WORDVEC_SIZE/4, (3,3), activation='relu')(image_global)
        image_global = layers.Conv2D(WORDVEC_SIZE/2, (3,3), activation='relu')(image_global)
        image_global = layers.Flatten()(image_global)

    image_global = layers.Concatenate()([image_global, ctx])
    image_global = layers.Dense(1024, activation='relu')(image_global)

    image_global = layers.BatchNormalization()(image_global)
    image_global = layers.Dense(WORDVEC_SIZE/2, activation=ACTIVATION)(image_global)
    image_global = layers.BatchNormalization()(image_global)
    image_global = layers.RepeatVector(max_words)(image_global)

    # Local Image featuers (convnet output for just the bounding box)
    input_img_local = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    image_local = cnn(input_img_local)

    if USE_CGRU:
        image_local = layers.Conv2D(CGRU_SIZE, (1,1), padding='same', activation='relu')(image_local)
        res_cgru = SpatialCGRU(image_local, CGRU_SIZE)
        image_local = layers.add([image_local, res_cgru])

    if REDUCE_MEAN:
        image_local = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(image_local)
        image_local = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(image_local)
    else:
        image_local = layers.Conv2D(WORDVEC_SIZE/4, (3,3), activation='relu')(image_local)
        image_local = layers.Conv2D(WORDVEC_SIZE/2, (3,3), activation='relu')(image_local)
        image_local = layers.Flatten()(image_local)

    image_local = layers.Concatenate()([image_local, ctx])
    image_local = layers.Dense(1024, activation='relu')(image_local)

    image_local = layers.BatchNormalization()(image_local)
    image_local = layers.Dense(WORDVEC_SIZE/2, activation=ACTIVATION)(image_local)
    image_local = layers.BatchNormalization()(image_local)
    image_local = layers.RepeatVector(max_words)(image_local)

    language_model = models.Sequential()

    input_words = layers.Input(batch_shape=(BATCH_SIZE, max_words), dtype='int32')
    language = layers.Embedding(words.VOCABULARY_SIZE, WORDVEC_SIZE, input_length=max_words)(input_words)


    x = layers.concatenate([image_global, image_local, repeat_ctx, language])
    if RNN_TYPE == 'LSTM':
        x = layers.LSTM(RNN_SIZE)(x)
    else:
        x = layers.GRU(RNN_SIZE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(words.VOCABULARY_SIZE, activation='softmax')(x)

    return models.Model(inputs=[input_img_global, input_img_local, input_words, input_ctx], outputs=x)


# TODO: Move batching out to the generic runner
def training_generator(**params):
    max_words = params['max_words']
    while True:
        X_global = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_local = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_words = np.zeros((BATCH_SIZE, max_words), dtype=int)
        X_ctx = np.zeros((BATCH_SIZE,5))
        Y = np.zeros((BATCH_SIZE, words.VOCABULARY_SIZE))
        for i in range(BATCH_SIZE):
            x, y, box = process(*dataset_grefexp.example(), **params)
            x_global, x_local, x_words, x_ctx = x
            X_global[i] = x_global
            X_local[i] = x_local
            X_words[i] = x_words
            X_ctx[i] = x_ctx
            Y[i] = y
        yield [X_global, X_local, X_words, X_ctx], Y


# TODO: merge back into training_generator for sanity
def pg_training_generator(**params):
    max_words = params['max_words']
    while True:
        X_global = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_local = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_words = np.zeros((BATCH_SIZE, max_words), dtype=int)
        X_ctx = np.zeros((BATCH_SIZE,5))
        Y = np.zeros((BATCH_SIZE, words.VOCABULARY_SIZE))
        reference_texts = []
        for i in range(BATCH_SIZE):
            jpg_data, bbox, reference_text = dataset_grefexp.example()
            x, y, box = process(jpg_data, bbox, reference_text, **params)
            reference_texts.append(reference_text)
            x_global, x_local, x_words, x_ctx = x
            X_global[i] = x_global
            X_local[i] = x_local
            X_words[i] = x_words
            X_ctx[i] = x_ctx
            Y[i] = y
        yield [X_global, X_local, X_words, X_ctx], Y, reference_texts


def validation_generator(**params):
    for k in sorted(dataset_grefexp.get_all_keys()):
        jpg_data, box, texts = dataset_grefexp.get_annotation_for_key(k)
        x, y, box = process(jpg_data, box, texts, **params)
        x_global, x_local, x_words, x_ctx = x
        yield x_global, x_local, x_ctx, box, texts


def process(jpg_data, box, texts, **params):
    max_words = params['max_words']
    x_local = util.decode_jpg(jpg_data, crop_to_box=box)
    # hack: scale the box down
    x_global, box = util.decode_jpg(jpg_data, box)
    text = util.strip(random.choice(texts))
    indices = words.indices(text)
    idx = np.random.randint(1, len(indices))
    x_indices = indices[:idx]
    if len(x_indices) > max_words:
        x_indices = x_indices[-max_words:]

    x_indices = util.left_pad(x_indices, **params)
    y = util.onehot(indices[idx])

    x_ctx = img_ctx(box)
    return [x_global, x_local, x_indices, x_ctx], y, box


def img_ctx(box):
    x0, x1, y0, y1 = box
    left = float(x0) / IMG_WIDTH
    right = float(x1) / IMG_WIDTH
    top = float(y0) / IMG_HEIGHT
    bottom = float(y1) / IMG_HEIGHT
    box_area = float(x1 - x0) * (y1 - y0)
    img_area = IMG_HEIGHT * IMG_WIDTH
    x_ctx = np.array([left, top, right, bottom, box_area/img_area])
    return x_ctx


def evaluate(model, x_global, x_local, x_ctx, box, texts, verbose=True, **params):
    if verbose:
        img = x_global - x_global.min()
        util.show(x_local)
        util.show(img, box=box)
    candidate = predict(model, x_global, x_local, x_ctx, box, **params)
    candidate = util.strip(candidate)
    references = map(util.strip, texts)
    #print("{} {} ({})".format(likelihood, candidate, references[0]))
    return candidate, references


def get_scores(candidate_list, references_list):
    scores = {}
    scores['bleu1'] = np.mean([bleu(c, ref, n=1) for c, ref in zip(candidate_list, references_list)])
    scores['bleu2'] = np.mean([bleu(c, ref, n=2) for c, ref in zip(candidate_list, references_list)])
    scores['bleu4'] = np.mean([bleu(c, ref, n=4) for c, ref in zip(candidate_list, references_list)])
    scores['rouge'] = np.mean([rouge(c, ref) for c, ref in zip(candidate_list, references_list)])
    return scores


# Alternate implementation of BLEU, different than the nltk implementation
def bleu(candidate, references, n=4):
    weights = [1.0 / n] * n
    scores, _ = bleu_scorer.BleuScorer(candidate, references, n=n).compute_score(option='closest')
    return scores


def rouge(candidate, references):
    return rouge_scorer.Rouge().calc_score([candidate], references)


def predict(model, x_global, x_local, x_ctx, box, **params):
    max_words = params['max_words']
    # An entire batch must be run at once, but we only use the first slot in that batch
    indices = util.left_pad([words.START_TOKEN_IDX], **params)
    x_global = util.expand(x_global, BATCH_SIZE)
    x_local = util.expand(x_local, BATCH_SIZE)
    indices = util.expand(indices, BATCH_SIZE)
    x_ctx = util.expand(x_ctx, BATCH_SIZE)

    # Input is empty padding followed by start token
    output_words = []
    for i in range(1, max_words):
        preds = model.predict([x_global, x_local, indices, x_ctx])
        indices = np.roll(indices, -1, axis=1)
        indices[:, -1] = np.argmax(preds[:], axis=1)

    return words.words(indices[0])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
