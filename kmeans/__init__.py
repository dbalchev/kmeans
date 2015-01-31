# -*- coding: UTF-8 -*-
from collections import defaultdict
from copy import deepcopy
import random

from .distances import default_similarity
from .utils import Vectorizer, WeightedMap, merge


class Centroid(object):
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
        self.center = merge(*self.items)


def fill_initial_clusters(cores, corpus, similarity):
    """
    Функция създаваща началните стойности нa клъстерите,
    като всеки от тях представлява центроид и ще преизчисли
    центровете на всеки центроид преди да приключи.
    """
    clusters = {core[0]: Centroid(core[1]) for core in cores}
    for item in corpus:
        max_similarity = (0, None)
        for label, cluster in clusters.items():
            cur_similarity = similarity(cluster.center, item[1])
            if cur_similarity > max_similarity[0]:
                max_similarity = (cur_similarity, label)

        if max_similarity[1] is None:
            raise Exception('Similarity function not working correctly.')

        clusters[max_similarity[1]].items.append(item[1])

    for cluster in clusters.values():
        cluster.centralize()

    return clusters


def kmeans(n_clusters, text_seq, similarity=default_similarity):
    """
    Функция използваща k-means алогоритъм за клъстеризация на текстове
    text_seq трябва да са двойки (label, text_vector), където text_vector е
    вектор върнат от Vectorizer
    similarity = similarity or default_similarity
    """
    corpus        = [(label, WeightedMap(list(sorted(text)))) for label, text in text_seq]
    cores         = random.sample(corpus, n_clusters)
    clusters      = fill_initial_clusters(cores, corpus, similarity)
    changed       = True
    while changed:
        changed = False
        new_clusters = deepcopy(clusters)
        for label, cluster in clusters.items():
            for item in cluster.items:
                max_similarity = (similarity(item[1], cluster.center), None)
                for inn_label, inn_cluster in clusters.items():
                    cur_similarity = similarity(inn_cluster.center, item[1])
                    if cur_similarity > max_similarity[0]:
                        max_similarity = (cur_similarity, inn_label)
                        changed = True

                new_clusters[label].items.remove(item[1])
                new_clusters[max_similarity[1]].items.append(item[1])

        clusters = new_clusters
        for cluster in clusters.values():
            cluster.centralize()

    return clusters


def clusterize(n_clusters, filename_seq, similarity=default_similarity):
    vec = Vectorizer()
    return kmeans(n_clusters, ((filename, vec.vectorize_file(filename)) \
        for filename in filename_seq), similarity)
