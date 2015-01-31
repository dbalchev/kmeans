from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .stopwords import STOPWORDS
from math import log
from collections import defaultdict
stemmer = WordNetLemmatizer()


# за да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> wordnet
def stem(s):
    return stemmer.lemmatize(s)

# за да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> punkt
def tokenize(filename):
    with open(filename) as inp:
        return (stem(token) for token in word_tokenize(inp.read(-1)))

def count_duplicates(lst, indx):
    cnt = 1
    el = lst[indx]
    indx += 1
    while indx < len(lst) and lst[indx] == el:
        indx += 1
        cnt += 1
    return cnt

def merge(lh, rh):
    if len(lh) == 0:
        return rh
    if len(rh) == 0:
        return lh
    result = WeightedMap([])
    for key, weight in lh.items():
        result[key] = weight * 0.5
    for key, weight in rh.items():
        result[key] += weight * 0.5
    norm = 0
    for w in result.values():
        norm += w * w
    result.norm = norm
    return result

def self_information(wm):
    si = 0.0
    for p in wm.values():
        si -= p * log(p, 2)
    return si

class WeightedMap(defaultdict):
    def __init__(self, vec):
        self.norm = 0
        if len(vec) == 0:
            super().__init__(float)
        else:
            def gen():
                cu = 0
                denominator = len(vec)
                while cu < len(vec):
                    cnt = count_duplicates(vec, cu)
                    p = cnt / denominator
                    self.norm += p * p
                    yield vec[cu], p
                    cu += cnt
            super().__init__(float, gen())



class Prenum(dict):
    def __init__(self):
        super().__init__(self)
        self.__counter = 0

    def __missing__(self, key):
        new_num = self.__counter
        self.__counter += 1
        self[key] = new_num
        return new_num

class Vectorizer:
    def __init__(self):
        self.prenum = Prenum()
        self.stopwords = STOPWORDS

    def vectorize_seq(self, seq):
        return (self.prenum[token] for token in seq \
            if token.isalpha() and token not in self.stopwords)

    def vectorize_file(self, filename):
        return self.vectorize_seq(tokenize(filename))
