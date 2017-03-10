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

model = target.build_model()
if os.path.exists(model_filename):
    model.load_weights(model_filename)

g = target.validation_generator()
scores = {}
for args in g:
    for i in range(10):
        s = target.evaluate(model, *args, temperature=.1*i)
        print s
    exit()

for k in scores:
    from scipy import stats
    print("Score {} statistics:".format(k))
    pprint(stats.describe(scores[k]))
