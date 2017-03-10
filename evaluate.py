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

normal_bleu1 = []
sampled_bleu1 = []
normal_bleu2 = []
sampled_bleu2 = []
normal_rouge = []
sampled_rouge = []
for args in g:
    normal_score = target.evaluate(model, *args, temperature=0)
    normal_bleu1.append(normal_score['bleu1'])
    normal_bleu2.append(normal_score['bleu2'])
    normal_rouge.append(normal_score['rouge'])

    SAMPLES = 10
    sample = []
    max_likelihood = 0
    max_likelihood_sample = None
    for i in range(SAMPLES):
        s = target.evaluate(model, *args, temperature=.1*i)
        if s['likelihood'] > max_likelihood:
            max_likelihood_sample = s
            max_likelihood = s['likelihood']
    print "best sample: {} {}".format(max_likelihood, max_likelihood_sample)
    sampled_bleu1.append(max_likelihood_sample['bleu1'])
    sampled_bleu2.append(max_likelihood_sample['bleu2'])
    sampled_rouge.append(max_likelihood_sample['rouge'])

print("Number of Captions: {}".format(len(normal_bleu2)))
for (name, data) in [('normal BLEU1', normal_bleu1), ('normal BLEU2', normal_bleu2), ('normal ROUGE', normal_rouge), 
        ('sampled BLEU1', sampled_bleu1), ('sampled BLEU2', sampled_bleu2), ('sampled ROUGE', sampled_rouge)]:
    print '{} min/max/mean'.format(name)
    print np.array(data).min(), np.array(data).max(), np.array(data).mean()
