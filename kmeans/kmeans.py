# -*- coding: UTF-8 -*-
from itertools import chain
import random

from .distances import default_similarity
from .utils import WeightedMap, merge
from .vectorizers import Vectorizer


class Centroid:
    """
    Клас, който ще представлява всеки центроид направен във клъстерите,
    той ще се инициализира със ядрото което е подбрано от популацията,
    и след приключването на всяка стъпка на k-means алгоритъма ще позваме
    метода centralize за да се изчисли новият център на центроида чрез
    средно аритметично.
    """
    def __init__(self, center):
        self.center = center
        self.items  = []

    def centralize(self):
        self.center = merge(*[item.content for item in self.items])


class FileRepr:
    def __init__(self, file_label, weighted_map):
        self.name    = file_label
        self.content = weighted_map


class KMeans:
    DEFAULT_SIMILARITY_METHOD   = default_similarity
    DEFAULT_MAX_ITERATIONS      = 1000
    DEFAULT_MIN_TEXT_LEN        = 10
    DEFAULT_ITERATIONS_CALLBACK = lambda x: None # lambda x: print("Done in", x, "iterations.")

    def __init__(self, n_clusters, filename_seq,
                 similarity_method=DEFAULT_SIMILARITY_METHOD,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 min_text_len=DEFAULT_MIN_TEXT_LEN,
                 iterations_callback=DEFAULT_ITERATIONS_CALLBACK):
        self.n_clusters          = n_clusters
        self.similarity_method   = similarity_method
        self.max_iterations      = max_iterations
        self.min_text_len        = min_text_len
        self.iterations_callback = iterations_callback
        self.vec = Vectorizer()
        text_seq = ((filename, self.vec.vectorize_file(filename)) for filename in filename_seq)
        self.corpus = [FileRepr(label, WeightedMap.from_vec(list(sorted(text))))\
                        for label, text in text_seq if len(text) >= self.min_text_len]

    def get_clusters(self, cores, corpus, old_cluster=None):
        """
        Функция създаваща началните стойности нa клъстерите,
        като всеки от тях представлява центроид и ще преизчисли
        центровете на всеки центроид преди да приключи.
        """
        clusters = {core.name: Centroid(core.content) for core in cores}
        changed = False
        for item in corpus:
            max_similarity = (None, None)
            for label, cluster in clusters.items():
                cur_similarity = self.similarity_method(cluster.center, item.content)
                # if label == item.name:
                #     max_similarity = (cur_similarity, label)
                #     break
                if not max_similarity[0] or cur_similarity > max_similarity[0]:
                    max_similarity = (cur_similarity, label)

            if max_similarity[1] is None:
                raise Exception('Similarity function not working correctly.')

            if old_cluster and max_similarity[1] != old_cluster[item.name]:
                changed = True
            clusters[max_similarity[1]].items.append(item)

        for cluster in clusters.values():
            cluster.centralize()

        return clusters, changed

    def clusterize(self):
        """
        Функция използваща k-means алогоритъм за клъстеризация на текстове
        вектор върнат от Vectorizer
        similarity = similarity or default_similarity
        """
        cores       = random.sample(self.corpus, self.n_clusters)
        clusters, _ = self.get_clusters(cores, self.corpus)
        changed     = True
        iteration   = 0
        while changed and iteration <= self.max_iterations:
            old_cluster = dict(chain.from_iterable(\
                ((item.name, label) for item in cluster.items) for label, cluster in clusters.items()))
            clusters, changed = self.get_clusters([FileRepr(name, cluster.center) \
                for name, cluster in clusters.items()], self.corpus, old_cluster=old_cluster)
            iteration += 1

        self.iterations_callback(iteration)
        return clusters
