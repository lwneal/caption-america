import os
import sys
import time
import importlib

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

for X, Y in g:
    print target.predict(model, X)
