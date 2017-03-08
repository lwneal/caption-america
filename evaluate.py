import os
import sys
import time
import importlib
import numpy as np
from pprint import pprint

module_name = sys.argv[1]
module_name = module_name.rstrip('.py')
target = importlib.import_module(module_name)

model_filename = '{}.{}.h5'.format(module_name, int(time.time()))
if len(sys.argv) > 2:
    model_filename = sys.argv[2]

if os.path.exists(model_filename):
    from keras.models import load_model
    model = load_model(model_filename)
else:
    model = target.build_model()

g = target.validation_generator()
scores = {}
for X, Y in g:
    s = target.evaluate(model, X, Y)
    for k in s:
        if k not in scores:
            scores[k] = []
        scores[k].append(s[k])
    means = {k: np.mean(scores[k]) for k in scores}
    print('\n[K{} {}'.format(len(scores[k]), means)),

for k in scores:
    from scipy import stats
    print("Score {} statistics:".format(k))
    pprint(stats.describe(scores[k]))
