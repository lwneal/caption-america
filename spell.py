import os
import sys
import json
import random
import enchant

# Spell check only against words in the vocabulary
checker = enchant.request_pwl_dict('vocabulary.txt')
def spell(text):
    words = []
    for w in text.split():
        suggestions = checker.suggest(w)
        if suggestions:
            words.append(suggestions[0])
        else:
            #print("Unknown word: {}".format(w))
            words.append(w)
    return ' '.join(words)
