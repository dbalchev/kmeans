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
        self.filename_seq        = filename_seq
        self.similarity_method   = similarity_method
        self.max_iterations      = max_iterations
        self.min_text_len        = min_text_len
        self.iterations_callback = iterations_callback

    def get_clusters(cores, corpus, old_cluster=None):
        """
        Функция създаваща началните стойности нa клъстерите,
        като всеки от тях представлява центроид и ще преизчисли
        центровете на всеки центроид преди да приключи.
        """
        clusters = {core.name: Centroid(core.content) for core in cores}
        changed = False
        for item in corpus:
            max_similarity = (-1, None)
            for label, cluster in clusters.items():
                cur_similarity = self.similarity_method(cluster.center, item.content)
                if cur_similarity > max_similarity[0]:
                    max_similarity = (cur_similarity, label)

            if max_similarity[1] is None:
                raise Exception('Similarity function not working correctly.')

            if old_cluster and max_similarity[1] != old_cluster[item.name]:
                changed = True
            clusters[max_similarity[1]].items.append(item)

        for cluster in clusters.values():
            cluster.centralize()

        return clusters, changed

    def kmeans(self, text_seq):
        """
        Функция използваща k-means алогоритъм за клъстеризация на текстове
        text_seq трябва да са двойки (label, text_vector), където text_vector е
        вектор върнат от Vectorizer
        similarity = similarity or default_similarity
        """
        corpus      = [FileRepr(label, WeightedMap.from_vec(list(sorted(text))))\
                        for label, text in text_seq if len(text) >= self.min_text_len]
        cores       = random.sample(corpus, self.n_clusters)
        clusters, _ = get_clusters(cores, corpus)
        changed     = True
        iteration   = 0
        while changed and iteration <= self.max_iterations:
            old_cluster = dict(chain.from_iterable(\
                ((item.name, label) for item in cluster.items) for label, cluster in clusters.items()))
            clusters, changed = get_clusters([FileRepr(name, cluster.center) \
                for name, cluster in clusters.items()], corpus, old_cluster=old_cluster)
            iteration += 1

        self.iterations_callback(iteration)
        return clusters

    def clusterize(self):
        vec = Vectorizer()
        return kmeans(((filename, vec.vectorize_file(filename)) for filename in self.filename_seq))
