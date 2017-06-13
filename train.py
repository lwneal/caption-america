import os
import time
import gpumemory
from keras import models
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
    if model_filename == 'default_model':
        model_filename = 'model.caption.{}.h5'.format(int(time.time()))
    model = caption.build_model(**params)
    if os.path.exists(model_filename):
        model.load_weights(model_filename)
        # TODO: Use load_model to allow loaded architecture to differ from code
        # Fix problems with custom layers like CGRU
        #model = models.load_model(model_filename)

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], decay=.01)
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
    for _ in range(validation_count):
        validation_example = next(g)
        c, r = caption.evaluate(model, *validation_example, **params)
        print("Candidate: {}".format(c))
        print("Refs: {}".format(r))
        candidate_list.append(c)
        references_list.append(r)
        scores = caption.get_scores(candidate_list, references_list)
        for s in scores:
            print("{}\t{:.3f}".format(s, scores[s]))


def train_pg(**params):
    model_filename = params['model_filename']
    batch_size = params['batch_size']
    epochs = params['epochs']

    model = caption.build_model(**params)
    model.load_weights(model_filename)
    model.summary()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'], decay=.01, learning_rate=.001)

    tg = caption.pg_training_generator(**params)

    for i in range(epochs):
        for _ in range(100):
            pg_x, pg_y, rewards = generate_pg_example(model, tg, **params)
            losses = model.train_on_batch(pg_x, pg_y, sample_weight=rewards)
            print(losses)
        model.save(model_filename)
        validate(model, **params)


def generate_pg_example(model, training_gen, **params):
    batch_size = params['batch_size']
    explore_temp = 0.5

    # Start at a random word somewhere in a random training example
    x, y, reference_texts = next(training_gen)
    x_glob, x_loc, x_words, x_ctx = x
    # HACK: Include the end token as a word
    reference_texts = [[r + ' 001' for r in reflist] for reflist in reference_texts]

    print("{} ...").format(words.words(x_words[0]))

    # Baseline score: rollout with temperature 0
    prev_steps = x_words.shape[1]
    rollout_steps = 5
    rollout_words = np.zeros((batch_size, prev_steps + rollout_steps), dtype=int)
    rollout_words[:, :prev_steps] = x_words
    for i in range(prev_steps, prev_steps + rollout_steps):
        preds = model.predict([x_glob, x_loc, rollout_words, x_ctx])
        rollout_words[i] = np.argmax(preds, axis=1)
    print("{}").format(words.words(rollout_words[0]))


    """
    best_next_word = np.zeros(batch_size, dtype=int)
    baseline_words = np.roll(x_words, -1, axis=1)
    baseline_words[:, -1] = best_next_word
    baseline_candidates = [words.words(s).strip('0 ') for s in baseline_words]
    baseline_scores = get_scores(baseline_words, reference_texts, **params)
    best_scores = np.ones_like(baseline_scores) * -1
    """

    # Take a random action
    preds = model.predict([x_glob, x_loc, x_words, x_ctx])
    new_words[:, -1] = [caption.sample(s, temperature=explore_temp) for s in action_distribution]
    new_words = np.roll(x_words, -1, axis=1)

    print("{} ...").format(words.words(x_words[0]))
    for r in range(rollouts):
        action_distribution = model.predict([x_glob, x_loc, x_words, x_ctx])
        new_words = np.roll(x_words, -1, axis=1)
        new_words[:, -1] = [caption.sample(s, temperature=rollout_temp) for s in action_distribution]
        
        word = words.words(new_words[:1, -1])
        print('\t... {}'.format(word))

        # TODO: at this point, roll all the way out with low temperature
        # Evaluate each sentence with it's extra word
        bleu2_scores = get_scores(new_words, reference_texts, **params)
        for i in range(batch_size):
            if bleu2_scores[i] > best_scores[i]:
                best_scores[i] = bleu2_scores[i]
                best_next_word[i] = new_words[i, -1]

    # Display
    ml_words = words.words(np.argmax(y, axis=-1)).split()
    pg_words = words.words(best_next_word).split()
    rewards = best_scores - baseline_scores

    return x, best_next_word, rewards


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


def ints_to_words(x_words):
    return [clean_text(words.words(s)) for s in x_words]


def clean_text(s, include_end=False):
    s, _, _ = s.partition('001')
    s = s.lstrip('0 ').strip()
    if include_end:
        return s + ' 001'
    return s

