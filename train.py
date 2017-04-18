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

    model = caption.build_model(**params)
    if os.path.exists(model_filename):
        model.load_weights(model_filename)
        # TODO: Use load_model to allow loaded architecture to differ from code
        #model = models.load_model(model_filename)

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], decay=.01)
    tg = caption.training_generator(**params)
    for i in range(epochs):
        caption.demo(model)
        validate(model, **params)
        model.fit_generator(tg, batches_per_epoch)
        model.save(model_filename)
    print("Finished training {} epochs".format(epochs))


def validate(model, validation_count=5000, **params):
    g = caption.validation_generator()
    print("Validating on {} examples...".format(validation_count))
    candidate_list = []
    references_list = []
    for _ in range(validation_count):
        validation_example = next(g)
        c, r = caption.evaluate(model, *validation_example, **params)
        print("{} ({})".format(c, r))
        candidate_list.append(c)
        references_list.append(r)
        scores = caption.get_scores(candidate_list, references_list)
        print scores

