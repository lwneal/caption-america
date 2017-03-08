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
    model = models.load_model(model_filename)
else:
    model = target.build_model()

g = target.training_generator()

while True:
    model.fit_generator(g, samples_per_epoch=2**12, nb_epoch=1)
    model.save(model_filename)
    target.demo(model)
