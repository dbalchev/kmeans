# -*- coding: UTF-8 -*-

from itertools import chain
from functools import lru_cache
from collections import Counter, defaultdict
from math import log, sqrt
from weakref import ref
EPS = 1e-9
class TFIDFDataBase:
    DEFAULT_MTF_CACHE_SIZE  = 512 * 1024

    def __init__(self, corpus, *,
                 mtf_cache_size=DEFAULT_MTF_CACHE_SIZE):
        corpus = {frepr.name:frepr.content for frepr in corpus}

        self.mtf = lru_cache(maxsize=mtf_cache_size)(self._mtf_generate)
        self.idf = TFIDFDataBase._generate_idf(corpus)
        #TODO да си има собствен параметър за максимален размер
        self.dot_square = lru_cache(maxsize=mtf_cache_size)(self._dot_square)

    def tfidf(self, word_index, document):
        return self.tf(word_index, document) \
            * self.idf[word_index]

    def _mtf_generate(self, document):
        return max(document.values())

    def tf(self, word_index, document):
        return document[word_index] \
            / self.mtf(document)

    def _dot_square(self, document):
        result = 0
        for word in document.keys():
            t = self.tfidf(word, document)
            result += t * t
        return result

    def tfidf_dot(self, lh, rh):
        if len(lh) > len(rh):
            lh, rh = rh, lh
        result = 0
        for word in lh.keys():
            result += self.tfidf(word, lh) * self.tfidf(word, rh)
        return result

    def euclidean_similarity(self, lh, rh):
        # l_norm  = len(lh) / len(key_set)
        # r_norm  = len(rh) / len(key_set)
        lh_dot = self.dot_square(lh)
        rh_dot = self.dot_square(rh)
        norm = lh_dot - 2 * self.tfidf_dot(lh, rh) + rh_dot
        denominator = lh_dot * rh_dot
        if abs(denominator) < EPS:
            return 1
        if abs(norm) < EPS:
            norm = 0
        try:
            return 1 - sqrt(norm / denominator)
        except ValueError:
            print(norm, denominator)
            raise

    @staticmethod
    def _generate_idf(corpus):
        word_counter = Counter()
        for document in corpus.values():
            for word in document.keys():
                word_counter[word] += 1
        return {word:log(len(corpus) / freq, 2) \
            for word, freq in word_counter.items()}
