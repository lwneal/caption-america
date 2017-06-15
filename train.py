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

    pg = caption.pg_training_generator(**params)
    tg = caption.training_generator(**params)

    for i in range(epochs):
        for _ in range(batches_per_epoch):
            # Policy Gradient training
            pg_x, pg_y, rewards = generate_pg_example(model, pg, **params)
            losses = model.train_on_batch(pg_x, pg_y, sample_weight=rewards)
            # ML Training
            #x, y = next(tg)
            #losses = model.train_on_batch(x, y)
        model.save(model_filename)
        validate(model, **params)


def generate_pg_example(model, training_gen, **params):
    batch_size = params['batch_size']
    best_of_n = params['best_of_n']
    sample_temp = params['sample_temp']
    policy_steps = np.random.randint(params['max_policy_steps'])

    # Start at a random word somewhere in a random training example
    x, y, reference_texts = next(training_gen)
    x_glob, x_loc, x_words, x_ctx = x
    # HACK: Include the end token as a word
    reference_texts = [[r for r in reflist] for reflist in reference_texts]

    # Follow policy to get into a real-world state
    for _ in range(policy_steps):
        policy_preds = model.predict([x_glob, x_loc, x_words, x_ctx])
        x_words = np.roll(x_words, -1, axis=1)
        x_words[:, -1] = np.argmax(policy_preds, axis=1)

    # Sampled Score: score for the sentence plus one more word
    sampled_words = np.zeros((batch_size, x_words.shape[1] + 1), dtype=int)
    sampled_words[:, :-1] = x_words

    preds = model.predict(x)
    preds = np.argsort(preds, axis=1)
    top_n = preds[:, -best_of_n:]

    sampled_words[:, -1] = top_n[:, 0]
    #baseline_score = get_scores(sampled_words, reference_texts, **params)
    # HACK: try setting baseline score to constant
    baseline_score = 0.1

    best_score = np.zeros(batch_size)
    best_word = np.zeros(batch_size, dtype=int)
    for i in range(best_of_n):
        sampled_words[:, -1] = top_n[:, i]
        score = get_scores(sampled_words, reference_texts, **params)
        for j in range(batch_size):
            if score[j] > best_score[j]:
                best_score[j] = score[j]
                best_word[j] = top_n[j, i]

    reward = best_score - baseline_score

    # Display
    idx = np.random.randint(batch_size)
    true_words = words.words(x_words[idx, :-policy_steps])
    policy_words = words.words(x_words[idx, -policy_steps:])
    baseline_word = words.words(top_n[:,0]).split()[idx]
    chosen_word = words.words(best_word).split()[idx]
    print("{} {} {}/{} ({:+.3f})".format(
        true_words, cyan(policy_words), baseline_word, yellow(chosen_word), reward[idx]))

    return x, best_word, reward


def cyan(val):
    return '[36m{}[0m'.format(val)


def yellow(val):
    return '[33m{}[0m'.format(val)


def red(val):
    return '[35m{}[0m'.format(val)


def green(val):
    return '[32m{}[0m'.format(val)


def get_scores(x_words, refs, **params):
    candidates = ints_to_words(x_words, include_end=False)
    def bleu2():
        return np.array([caption.bleu(c, r)[1] for (c, r) in zip(candidates, refs)])
    def bleu4():
        return np.array([caption.bleu(c, r)[3] for (c, r) in zip(candidates, refs)])
    def rouge():
        return np.array([caption.rouge(c, r) for (c, r) in zip(candidates, refs)])
    if params['score'] == 'all':
        return (bleu2() + bleu4() + rouge()) / 3.0
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

