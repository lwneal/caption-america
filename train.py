import os
import time
import gpumemory
from keras import models
import caption

import os
import sys
import time
import importlib
import numpy as np
from pprint import pprint


def train(model_filename, epochs, batches_per_epoch, batch_size, **params):

    if model_filename == 'default_model':
        model_filename = 'model.caption.{}.h5'.format(int(time.time()))

    if os.path.exists(model_filename):
        print("Loading model from {}".format(model_filename))
        model = models.load_model(model_filename)
    else:
        print("Building new model")
        model = caption.build_model(**params)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], decay=.01)
    tg = caption.training_generator(**params)
    for i in range(epochs):
        caption.demo(model)
        validate(model, **params)
        model.fit_generator(tg, batches_per_epoch)
        model.save(model_filename)
    print("Finished training {} epochs".format(epochs))


def validate(model, validation_count=5000, **kwargs):
    g = caption.validation_generator()
    bleu1 = []
    bleu2 = []
    rouge = []
    print("Validating on {} examples...".format(validation_count))
    candidate_list = []
    references_list = []
    for _ in range(validation_count):
        validation_example = next(g)
        c, r = caption.evaluate(model, *validation_example)
        print c
        candidate_list.append(c)
        references_list.append(r)
    scores = caption.get_scores(candidate_list, references_list)
    print scores

