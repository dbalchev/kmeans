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
        self.dot_square = lru_cache(maxsize=mtf_cache_size) \
            (self._dot_square)
        self.tfidf_sum = lru_cache(maxsize=mtf_cache_size) \
            (self._tfidf_sum)

    def tfidf(self, word_index, document):
        return self.tf(word_index, document) \
            * self.idf[word_index]

    def _tfidf_sum(self, document):
        return sum(self.tfidf(word, document) \
            for word in document.keys())

    def _mtf_generate(self, document):
        return max(document.values())

    def tf(self, word_index, document):
        if len(document) == 0:
            return 0.5
        return 0.5 + 0.5 * document[word_index] \
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
        rh_tfidf_sum = self.tfidf_sum(rh)
        for word in lh.keys():
            rh_tfidf = self.tfidf(word, rh)
            rh_tfidf_sum -= rh_tfidf
            result += self.tfidf(word, lh) * rh_tfidf
        print("\n", rh_tfidf_sum)
        result += 0.5 * rh_tfidf_sum
        return result

    def euclidean_similarity(self, lh, rh):
        # l_norm  = len(lh) / len(key_set)
        # r_norm  = len(rh) / len(key_set)
        lh_dot = self.dot_square(lh)
        rh_dot = self.dot_square(rh)
        if not lh_dot:
            if not rh_dot:
                return 1
            return 0
        if not rh_dot:
            return 0
        norm = lh_dot - 2 * self.tfidf_dot(lh, rh) + rh_dot
        denominator = lh_dot * rh_dot
        if abs(denominator) < EPS:
            return 1
        if abs(norm) < EPS:
            norm = 0
        try:
            return 1 - sqrt(norm / denominator)
        except ValueError:
            print()
            print(lh)
            print(rh)
            print(norm, denominator)
            raise

    def kl_similarity(self, lh, rh):
        if not lh and not rh:
            return 1
        slen = len(lh) + len(rh)
        cl, cr = (len(x) / slen for x in (lh, rh))
        key_set = set(chain(lh.keys(), rh.keys()))

        result = 0
        for key in key_set:
            lw = self.tfidf(key, lh)
            rw = self.tfidf(key, rh)
            denominator = lw + rw
            result += cl * TFIDFDataBase._d_kl(lw, denominator)\
                + cr * TFIDFDataBase._d_kl(rw, denominator)

        return result
    @staticmethod
    def _d_kl(wa, wb):
        if abs(wa) < EPS:
            return 0
        return wa * log(wa / wb, 2)
    @staticmethod
    def _generate_idf(corpus):
        word_counter = Counter()
        for document in corpus.values():
            for word in document.keys():
                word_counter[word] += 1
        return {word:log(len(corpus) / freq, 2) \
            for word, freq in word_counter.items()}
