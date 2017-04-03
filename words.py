import re

# NOTE: Vocabulary must contain the start and end tokens at these positions
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


# Remove punctuation 
pattern = re.compile('[\W_]+')
def process(text):
    text = text.lower()
    text = pattern.sub(' ', text.lower())
    return text


def indices(text):
    text = process(text)
    wordlist = ('000 ' + text + ' 001').split()
    return [vocab.get(w, UNKNOWN_IDX) for w in wordlist]
