# -*- coding: UTF-8 -*-
from collections import defaultdict
from copy import deepcopy
from itertools import chain
import random

from .distances import default_similarity
from .utils import Vectorizer, WeightedMap, merge

MIN_TEXT_LEN = 10
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


def get_initial_clusters(cores, corpus, old_cluster, similarity):
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
            cur_similarity = similarity(cluster.center, item.content)
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


def kmeans(n_clusters, text_seq, similarity=default_similarity):
    """
    Функция използваща k-means алогоритъм за клъстеризация на текстове
    text_seq трябва да са двойки (label, text_vector), където text_vector е
    вектор върнат от Vectorizer
    similarity = similarity or default_similarity
    """
    corpus         = [FileRepr(label, WeightedMap.from_vec(list(sorted(text)))) for label, text in text_seq if len(text) >= MIN_TEXT_LEN]
    cores          = random.sample(corpus, n_clusters)
    clusters, _    = get_initial_clusters(cores, corpus, None, similarity)
    changed        = True
    iteration      = 0
    max_iterations = 1000
    while changed and iteration <= max_iterations:

        # if iteration == max_iterations:
        #     print('======================')
        #     for cluster_name, cluster in clusters.items():
        #     	print('Cluster Name:',cluster_name)
        #     	for item in cluster.items:
        #     		print('\t',item.name)

        old_cluster = dict(chain.from_iterable(\
            ((item.name, label) for item in cluster.items) for label, cluster in clusters.items()))
        clusters, changed = get_initial_clusters([FileRepr(name, cluster.center) \
            for name, cluster in clusters.items()], corpus, old_cluster, similarity)
        iteration += 1
    print("done in", iteration, "iters")
    return clusters


def clusterize(n_clusters, filename_seq, similarity=default_similarity):
    vec = Vectorizer()
    return kmeans(n_clusters, ((filename, vec.vectorize_file(filename)) \
        for filename in filename_seq), similarity)
