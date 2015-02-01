# -*- coding: UTF-8 -*-

from itertools import chain
from functools import lru_cache
from collections import Counter, defaultdict
from math import log, sqrt
from weakref import ref

class TFIDFDataBase:
    DEFAULT_MTF_CACHE_SIZE  = 512 * 1024

    def __init__(self, corpus, *,
                 mtf_cache_size=DEFAULT_MTF_CACHE_SIZE):
        corpus = {frepr.name:frepr.content for frepr in corpus}

        self.mtf = lru_cache(maxsize=mtf_cache_size)(self._mtf_generate)
        self.idf = TFIDFDataBase._generate_idf(corpus)

    def tfidf(self, word_index, document):
        return self.tf(word_index, document) \
            * self.idf[word_index]

    def _mtf_generate(self, document):
        return max(document.values())

    def tf(self, word_index, document):
        return 0.5 + 0.5 * document[word_index] \
            / self.mtf(document)

    def euclidean_similarity(self, lh, rh):
        sq  = lambda x: x * x
        key_set = set(lh.keys()) & set(rh.keys())
        # l_norm  = len(lh) / len(key_set)
        # r_norm  = len(rh) / len(key_set)
        return 1 - sqrt(sum(sq(self.tfidf(k, lh) - self.tfidf(k, rh)) for k in key_set))

    @staticmethod
    def _generate_idf(corpus):
        word_counter = Counter()
        for document in corpus.values():
            for word in document.keys():
                word_counter[word] += 1
        return {word:log(len(corpus) / freq, 2) \
            for word, freq in word_counter.items()}
