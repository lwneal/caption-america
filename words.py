UNKNOWN_IDX = 1

words = open('vocabulary.txt').read().split()
VOCABULARY_SIZE = len(words)
vocab = {}
for i, word in enumerate(words):
    vocab[i] = word
    vocab[word] = i


def words(indices):
    return ' '.join(vocab[i] for i in indices)


def indices(text):
    wordlist = ('000 ' + text + ' 001').lower().split()
    return [vocab.get(w, UNKNOWN_IDX) for w in wordlist]
