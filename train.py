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
    if params['training_mode'] == 'maximum-likelihood':
        train_ml(**params)
    else:
        train_pg(**params)


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


def validate(model, validation_count=5000, **params):
    g = caption.validation_generator(**params)
    print("Validating on {} examples...".format(validation_count))
    candidate_list = []
    references_list = []
    for _ in range(validation_count):
        validation_example = next(g)
        c, r = caption.evaluate(model, *validation_example, **params)
        print("{} ({})".format(c, r))
        candidate_list.append(c)
        references_list.append(r)
        scores = caption.get_scores(candidate_list, references_list)
        print scores


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
    horizon = np.random.randint(1)
    rollouts = 10
    sampling_temperature = 0.3
    x, y, reference_texts = next(training_gen)

    # HACK: Count the end token
    reference_texts = [[r + ' 001' for r in reflist] for reflist in reference_texts]

    # Roll out N random trajectories
    # For each one, get a BLEU-2 score
    # Choose the one with the highest BLEU-2
    # Turn it into a sequence of training examples!
    x_glob, x_loc, x_words, x_ctx = x

    # Wander outside of the training set
    for _ in range(horizon):
        x_words = np.roll(x_words, -1, axis=1)
        action_distribution = model.predict([x_glob, x_loc, x_words, x_ctx])
        x_words[:, -1] = [caption.sample(s, temperature=.0) for s in action_distribution]

    # Now what's the best word? Probably not the ground truth; that ship has sailed
    # By now we've output a few of our own words
    # Let's decide on the best word by randomly trying a few
    best_next_word = np.zeros(batch_size, dtype=int)
    baseline_words = np.roll(x_words, -1, axis=1)
    baseline_words[:, -1] = best_next_word
    baseline_candidates = [words.words(s).strip('0 ') for s in baseline_words]
    baseline_scores = get_scores(baseline_words, reference_texts)
    best_scores = np.ones_like(baseline_scores) * -1

    for r in range(rollouts):
        action_distribution = model.predict([x_glob, x_loc, x_words, x_ctx])
        new_words = np.roll(x_words, -1, axis=1)
        new_words[:, -1] = [caption.sample(s, temperature=sampling_temperature) for s in action_distribution]

        # Evaluate each sentence with it's extra word
        bleu2_scores = get_scores(new_words, reference_texts)

        for i in range(batch_size):
            if bleu2_scores[i] > best_scores[i]:
                #print('"{}" raises score from {:.2f} to {:.2f}'.format(words.words(new_words[i]).strip('0 '), best_scores[i], bleu2_scores[i]))
                best_scores[i] = bleu2_scores[i]
                best_next_word[i] = new_words[i, -1]

    # Display
    ml_words = words.words(np.argmax(y, axis=-1)).split()
    pg_words = words.words(best_next_word).split()
    rewards = best_scores - baseline_scores
    for i in range(batch_size):
        print("{} ... {} ({:.2f}) {}".format(words.words(x_words[i]), pg_words[i], rewards[i], reference_texts[i]))

    return x, best_next_word, rewards


def get_scores(x_words, refs):
    candidates = [words.words(s).replace('001', '').replace('0', '').strip(' ') + ' 001' for s in x_words]
    bleu2 = np.array([caption.bleu(c, r)[1] for (c, r) in zip(candidates, refs)])
    rouge = np.array([caption.rouge(c, r) for (c, r) in zip(candidates, refs)])
    return bleu2 + rouge


