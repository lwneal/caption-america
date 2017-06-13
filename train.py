import os
import time
import gpumemory
from keras import models, optimizers
import caption

import os
import sys
import time
import importlib
import numpy as np
import words
from pprint import pprint


def train(**params):
    if params['mode'] == 'maximum-likelihood':
        train_ml(**params)
    elif params['mode'] == 'policy-gradient':
        train_pg(**params)
    else:
        validate(**params)


def train_ml(model_filename, epochs, batches_per_epoch, batch_size, **params):
    decay = params['decay']
    learning_rate = params['learning_rate']

    if model_filename == 'default_model':
        model_filename = 'model.caption.{}.h5'.format(int(time.time()))
    model = caption.build_model(**params)
    if os.path.exists(model_filename):
        model.load_weights(model_filename)
        # TODO: Use load_model to allow loaded architecture to differ from code
        # Fix problems with custom layers like CGRU
        #model = models.load_model(model_filename)

    model.summary()
    opt = optimizers.Adam(decay=decay, lr=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    tg = caption.training_generator(**params)
    for i in range(epochs):
        validate(model, **params)
        model.fit_generator(tg, batches_per_epoch)
        model.save(model_filename)
    print("Finished training {} epochs".format(epochs))


def validate(model=None, **params):
    validation_count = params['validation_count']

    if model == None:
        model = caption.build_model(**params)
        model.load_weights(params['model_filename'])

    g = caption.validation_generator(**params)
    print("Validating on {} examples...".format(validation_count))
    candidate_list = []
    references_list = []
    for i in range(validation_count):
        validation_example = next(g)
        c, r = caption.evaluate(model, *validation_example, **params)
        candidate_list.append(c)
        references_list.append(r)
        scores = caption.get_scores(candidate_list, references_list)
        print("{}/{}".format(i+1, validation_count))
        print("Candidate: {}".format(c))
        print("Refs: {}".format(r))
        for s in scores:
            print("{}\t{:.3f}".format(s, scores[s]))


def train_pg(**params):
    batches_per_epoch = params['batches_per_epoch']
    model_filename = params['model_filename']
    batch_size = params['batch_size']
    epochs = params['epochs']
    decay = params['decay']
    learning_rate = params['learning_rate']

    model = caption.build_model(**params)
    model.load_weights(model_filename)
    model.summary()
    opt = optimizers.Adam(decay=decay, lr=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    tg = caption.pg_training_generator(**params)

    for i in range(epochs):
        for _ in range(batches_per_epoch):
            pg_x, pg_y, rewards = generate_pg_example(model, tg, **params)
            losses = model.train_on_batch(pg_x, pg_y, sample_weight=rewards)
            print(losses)
        model.save(model_filename)
        validate(model, **params)


def generate_pg_example(model, training_gen, **params):
    batch_size = params['batch_size']
    best_of_n = params['best_of_n']
    sample_temp = 1.0

    # Start at a random word somewhere in a random training example
    x, y, reference_texts = next(training_gen)
    x_glob, x_loc, x_words, x_ctx = x
    # HACK: Include the end token as a word
    reference_texts = [[r + ' 001' for r in reflist] for reflist in reference_texts]

    print("{} ...").format(words.words(x_words[0]))

    # Baseline Score: rollout with temperature 0
    prev_steps = x_words.shape[1]
    rollout_steps = 5
    baseline_rollout = np.zeros((batch_size, prev_steps + rollout_steps), dtype=int)
    baseline_rollout[:, :prev_steps] = x_words
    for i in range(prev_steps, prev_steps + rollout_steps):
        preds = model.predict([x_glob, x_loc, baseline_rollout[:, i - prev_steps: i], x_ctx])
        baseline_rollout[:, i] = np.argmax(preds, axis=1)
    baseline_score = get_scores(baseline_rollout, reference_texts, **params)
    print("\t{} ({:.4f})").format(words.words(baseline_rollout[0]), baseline_score[0])


    def rollout_sample(sampled_word):
        # Roll out the sampled predictions to compare them with the baseline
        sample_rollout = np.zeros((batch_size, prev_steps + rollout_steps), dtype=int)
        sample_rollout[:, :prev_steps] = x_words
        sample_rollout[:, prev_steps] = sampled_word
        for i in range(prev_steps + 1, prev_steps + rollout_steps):
            preds = model.predict([x_glob, x_loc, sample_rollout[:, i - prev_steps:i], x_ctx])
            sample_rollout[:, i] = np.argmax(preds, axis=1)
        sample_score = get_scores(sample_rollout, reference_texts, **params)
        print("\t{} ({:+.4f})").format(words.words(sample_rollout[0]), sample_score[0] - baseline_score[0])
        return sample_score

    # Sample Score
    sample_preds = model.predict([x_glob, x_loc, x_words, x_ctx])
    best_scores = np.zeros(batch_size)
    best_words = np.zeros(batch_size, dtype=int)
    for _ in range(best_of_n):
        sampled_word = [caption.sample(p, temperature=sample_temp) for p in sample_preds]
        sample_score = rollout_sample(sampled_word) - baseline_score
        for i in range(batch_size):
            if sample_score[i] > best_scores[i]:
                best_words[i] = sampled_word[i]
                best_scores[i] = sample_score[i]

    return x, best_words, best_scores


def get_scores(x_words, refs, **params):
    candidates = ints_to_words(x_words, include_end=True)
    def bleu2():
        return np.array([caption.bleu(c, r)[1] for (c, r) in zip(candidates, refs)])
    def bleu4():
        return np.array([caption.bleu(c, r)[3] for (c, r) in zip(candidates, refs)])
    def rouge():
        return np.array([caption.rouge(c, r) for (c, r) in zip(candidates, refs)])
    if params['score'] == 'all':
        return bleu2() + bleu4() + rouge()
    elif params['score'] == 'bleu2':
        return bleu2()
    elif params['score'] == 'bleu4':
        return bleu4()
    elif params['score'] == 'rouge':
        return rouge()


def ints_to_words(x_words, **kwargs):
    return [clean_text(words.words(s), **kwargs) for s in x_words]


def clean_text(s, include_end=False):
    s, _, _ = s.partition('001')
    s = s.lstrip('0 ').strip()
    if include_end:
        return s + ' 001'
    return s

