import os
import sys
import time
import importlib

module_name = sys.argv[1]
module_name = module_name.rstrip('.py')
target = importlib.import_module(module_name)

model_filename = 'model.{}.{}.h5'.format(module_name, int(time.time()))
if len(sys.argv) > 2:
    model_filename = sys.argv[2]


if os.path.exists(model_filename):
    from keras.models import load_model
    model = load_model(model_filename)
else:
    model = target.build_model()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], decay=.001)

g = target.training_generator()

i = 0
while True:
    model.fit_generator(g, samples_per_epoch=2**12, nb_epoch=1)
    i += 1
    print("After training {}k samples:".format(4 * i))
    model.save(model_filename)
    target.demo(model)
    if i == 3600:
        break
print("Training complete. BLEU-2 score should be ~.136")
