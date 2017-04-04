import os
import time
import gpumemory  # Import more memory
from keras import models, layers

import os
import sys
import time
import importlib
import numpy as np
from pprint import pprint

def get_params():
    args = docopt(__doc__)
    return {argname(k): argval(args[k]) for k in args}


def argname(k):
    return k.strip('<').strip('>').strip('--').replace('-', '_')


def argval(val):
    if hasattr(val, 'lower') and val.lower() in ['true', 'false']:
        return val.lower().startswith('t')
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def train(module_name, model_filename, epochs, batches_per_epoch, batch_size, **kwargs):
    target = __import__(module_name.rstrip('.py'))

    if model_filename == 'default_model':
        model_filename = 'model.{}.{}.h5'.format(module_name, int(time.time()))

    if os.path.exists(model_filename):
        print("Loading model from {}".format(model_filename))
        model = models.load_model(model_filename)
    else:
        print("Building new model")
        model = target.build_model(**kwargs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], decay=.01)
    tg = target.training_generator()
    for i in range(epochs):
        target.demo(model)
        validate(target, model)
        model.fit_generator(tg, batches_per_epoch)
        model.save(model_filename)
    print("Finished training {} epochs".format(epochs))


def validate(target, model):
    g = target.validation_generator()
    bleu1 = []
    bleu2 = []
    rouge = []
    print("Validating on 100 examples...")
    candidate_list = []
    references_list = []
    for _ in range(10):
        validation_example = next(g)
        c, r = target.evaluate(model, *validation_example)
        candidate_list.append(c)
        references_list.append(r)
    scores = target.get_scores(candidate_list, references_list)
    print scores


if __name__ == '__main__':
    params = get_params()
    train(**params)

