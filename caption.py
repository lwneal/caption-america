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
from util import MAX_WORDS
from util import IMG_HEIGHT, IMG_WIDTH, IMG_SHAPE, IMG_CHANNELS
from keras import applications

from cgru import SpatialCGRU

BATCH_SIZE = 16
LEARNABLE_CNN_LAYERS = 1

def build_model(GRU_SIZE=1024, WORDVEC_SIZE=300, ACTIVATION='relu', **kwargs):
    #cnn = applications.vgg16.VGG16(include_top=False)
    cnn = applications.resnet50.ResNet50(include_top=False)
    for layer in cnn.layers[:-LEARNABLE_CNN_LAYERS]:
        layer.trainable = False

    # Global Image featuers (convnet output for the whole image)
    input_img_global = layers.Input(batch_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    image_global = cnn(input_img_global)
    #image_global = SpatialCGRU(image_global, 256)
    #image_global = SpatialCGRU(image_global, 256)
    image_global = layers.Flatten()(image_global)
    image_global = layers.Dense(1024, activation='relu')(image_global)

    image_global = layers.BatchNormalization()(image_global)
    image_global = layers.Dense(WORDVEC_SIZE/2, activation=ACTIVATION)(image_global)
    image_global = layers.BatchNormalization()(image_global)
    image_global = layers.RepeatVector(MAX_WORDS)(image_global)

    # Context Vector input
    # normalized to [0,1] the values:
    # left, top, right, bottom, (box area / image area)
    input_ctx = layers.Input(batch_shape=(BATCH_SIZE, 5))
    ctx = layers.BatchNormalization()(input_ctx)
    ctx = layers.RepeatVector(MAX_WORDS)(ctx)

    language_model = models.Sequential()

    input_words = layers.Input(batch_shape=(BATCH_SIZE, MAX_WORDS), dtype='int32')
    language = layers.Embedding(words.VOCABULARY_SIZE, WORDVEC_SIZE, input_length=MAX_WORDS)(input_words)
    language = layers.BatchNormalization()(language)
    language = layers.GRU(GRU_SIZE, return_sequences=True)(language)
    language = layers.BatchNormalization()(language)
    language = layers.TimeDistributed(layers.Dense(WORDVEC_SIZE, activation=ACTIVATION))(language)
    language = layers.BatchNormalization()(language)

    # Problem with Keras 2: 
    # TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have types [uint8, uint8, bool, uint8] that don't all match.
    # Masking doesn't work along with concatenation.
    # How do I get mask_zero=True working in the embed layer?

    x = layers.concatenate([image_global, ctx, language])
    x = layers.GRU(GRU_SIZE, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(words.VOCABULARY_SIZE, activation='softmax')(x)

    return models.Model(inputs=[input_img_global, input_words, input_ctx], outputs=x)


# TODO: Move batching out to the generic runner
def training_generator():
    while True:
        X_global = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        #X_local = np.zeros((BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        X_words = np.zeros((BATCH_SIZE, MAX_WORDS), dtype=int)
        X_ctx = np.zeros((BATCH_SIZE,5))
        Y = np.zeros((BATCH_SIZE, MAX_WORDS, words.VOCABULARY_SIZE))
        for i in range(BATCH_SIZE):
            x, y = process(*dataset_grefexp.example())
            x_global, x_words, x_ctx = x
            X_global[i] = x_global
            #X_local[i] = x_local
            X_words[i] = x_words
            X_ctx[i] = x_ctx
            Y[i] = y
        yield [X_global, X_words, X_ctx], Y


def validation_generator():
    for k in dataset_grefexp.get_all_keys():
        jpg_data, box, texts = dataset_grefexp.get_annotation_for_key(k)
        x, y = process(jpg_data, box, texts)
        x_global, x_words, x_ctx = x
        yield x_global, x_ctx, box, texts


def process(jpg_data, box, texts):
    #x_local = util.decode_jpg(jpg_data, crop_to_box=box)
    # hack: scale the box down
    x_global, box = util.decode_jpg(jpg_data, box)
    text = util.strip(random.choice(texts))
    indices = words.indices(text)
    #idx = np.random.randint(0, len(indices) - 1)
    #indices = indices[idx:idx + MAX_WORDS - 1]
    indices = indices[:MAX_WORDS]

    x_words = util.right_pad(indices)
    y = util.onehot(util.right_pad(indices[1:]))

    x_ctx = img_ctx(box)
    return [x_global, x_words, x_ctx], y


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


def evaluate(model, x_global, x_ctx, box, texts):
    candidate = predict(model, x_global, x_ctx, box)
    candidate = util.strip(candidate)
    references = map(util.strip, texts)
    #print("{} {} ({})".format(likelihood, candidate, references[0]))
    return candidate, references


def get_scores(candidate_list, references_list):
    scores = {}
    from nltk.translate import bleu_score
    def no_smoothing(p_n, **kwargs):
        return p_n
    scores['nltk_bleu1'] = bleu_score.corpus_bleu(references_list, candidate_list, smoothing_function=no_smoothing, weights=[1.0])
    scores['nltk_bleu2'] = bleu_score.corpus_bleu(references_list, candidate_list, smoothing_function=no_smoothing, weights=[.5, .5])
    scores['alt_bleu1'] = np.mean([bleu(c, ref, n=1) for c, ref in zip(candidate_list, references_list)])
    scores['alt_bleu2'] = np.mean([bleu(c, ref, n=2) for c, ref in zip(candidate_list, references_list)])
    scores['rouge'] = np.mean([rouge(c, ref) for c, ref in zip(candidate_list, references_list)])
    return scores


# Alternate implementation of BLEU, different than the nltk implementation
def bleu(candidate, references, n=4):
    weights = [1.0 / n] * n
    scores, _ = bleu_scorer.BleuScorer(candidate, references, n=2).compute_score(option='closest')
    return scores


def rouge(candidate, references):
    return rouge_scorer.Rouge().calc_score([candidate], references)



def predict(model, x_global, x_ctx, box):
    # An entire batch must be run at once, but we only use the first slot in that batch
    indices = util.left_pad([])
    x_global = util.expand(x_global, BATCH_SIZE)
    indices = util.expand(indices, BATCH_SIZE)
    x_ctx = util.expand(x_ctx, BATCH_SIZE)

    # Input is empty padding
    preds = model.predict([x_global, indices, x_ctx])[0]
    return words.words(np.argmax(preds, axis=1))


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)



def demo(model):
    for f in ['cat.jpg', 'dog.jpg', 'horse.jpg', 'car.jpg']:
        x_global = util.decode_jpg(f)
        height, width, _ = x_global.shape
        box = (width * .1, width * .9, height * .1, height * .9)
        #x_local = util.decode_jpg(f, crop_to_box=box)
        x_ctx = img_ctx(box)
        print("Prediction for {} {}:".format(f, box)),
        print(predict(model, x_global, x_ctx, box))
