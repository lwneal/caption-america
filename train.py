import sys
import os
from keras import models, layers
import resnet50
import numpy as np
from PIL import Image

MAX_WORDS = 20
VOCABULARY_SIZE = 100

def build_model():
    resnet = resnet50.ResNet50(include_top=True)
    for layer in resnet.layers:
        layer.trainable = False

    # image output: 1000-dim per word
    image_model = models.Sequential()
    image_model.add(resnet)
    image_model.add(layers.RepeatVector(MAX_WORDS))

    language_model = models.Sequential()
    language_model.add(layers.Embedding(VOCABULARY_SIZE, 300, input_length=MAX_WORDS, mask_zero=True))
    language_model.add(layers.GRU(256, return_sequences=True))
    language_model.add(layers.TimeDistributed(layers.Dense(128)))
    
    model = models.Sequential()
    model.add(layers.Merge([image_model, language_model], mode='concat', concat_axis=-1))
    model.add(layers.GRU(256, return_sequences=False))
    model.add(layers.Dense(VOCABULARY_SIZE))
    model.add(layers.Activation('softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def img(name):
    pil_img = Image.open(name).resize((224,224))
    pil_img.load()
    return np.array(pil_img) / 255.


def onehot(index):
    res = np.zeros(VOCABULARY_SIZE)
    res[index] = 1.0
    return res


def left_pad(indices):
    res = np.zeros(MAX_WORDS, dtype=int)
    res[MAX_WORDS - len(indices):] = indices
    return res


def expand(x):
    return np.expand_dims(x, axis=0)

def predict(model, img):
    words = left_pad([])
    for _ in range(MAX_WORDS):
        preds = model.predict([img, expand(words)])
        words = np.roll(words, -1)
        words[-1] = np.argmax(preds[0], axis=-1)
    return words

def example():
    words = range(1,MAX_WORDS+1)
    idx = np.random.randint(0, MAX_WORDS - 1)
    x_words = expand(left_pad(words[:idx]))
    y = expand(onehot(words[idx]))
    return [cat, x_words], y

def gen():
    while True: yield example()

model = build_model()
cat = expand(img('cat.jpg'))

for _ in range(1000):
    x, y = example()
    model.train_on_batch(x, y)
    print predict(model, cat)
