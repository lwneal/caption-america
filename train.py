import os
import sys
import time
import importlib

# The training set contains 50k sentences
# Each sentence contains ~10 words
# One epoch should be around 500k, or ~100 iterations
iter_count = 300

module_name = sys.argv[1]
module_name = module_name.rstrip('.py')
target = importlib.import_module(module_name)

model_filename = 'model.{}.{}.h5'.format(module_name, int(time.time()))
if len(sys.argv) > 2:
    model_filename = sys.argv[2]

model = target.build_model()
if os.path.exists(model_filename):
    model.load_weights(model_filename)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], decay=.01, lr=.001)

g = target.training_generator()

for i in range(iter_count):
    samples = 2**12
    print("Trained {}k samples:".format(i * samples / 2**10))
    target.demo(model)
    model.fit_generator(g, samples_per_epoch=samples, nb_epoch=1)
    model.save_weights(model_filename)
