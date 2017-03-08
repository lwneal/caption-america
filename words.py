START_TOKEN_IDX = 2
END_TOKEN_IDX = 3

words = open('vocabulary.txt').read().split()
vocab = {}
for i, word in enumerate(words):
    vocab[i] = word
    vocab[word] = i

VOCABULARY_SIZE = len(words)
UNKNOWN_IDX = vocab['thing']


def words(indices):
    return ' '.join(vocab[i] for i in indices)


def indices(text):
    wordlist = ('000 ' + text + ' 001').lower().split()
    return [vocab.get(w, UNKNOWN_IDX) for w in wordlist]
