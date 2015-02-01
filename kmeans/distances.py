from itertools import chain
from math import sqrt

from .utils import count_duplicates, self_information, merge, WeightedMap

def dot_product(lh, rh):
    if len(lh) > len(rh):
        lh, rh = rh, lh
    result = 0
    for l_word, l_weight in lh.items():
        result += l_weight * rh[l_word]
    return result

def cosine_distance(lh, rh):
    """
    Прави нормализирано косинусово разстояние на lh и rh;
    lh и rh трябва да са WeightedMap-ове
    """
    return 1 - cosine_similarity(lh, rh)


def cosine_similarity(lh, rh):
    """
    Прави нормализирано косинусово подобие на lh и rh;
    lh и rh трябва да са WeightedMap-ове
    """
    if not isinstance(lh, WeightedMap):
        raise ValueError("lh is not a WeightedMap")
    if not isinstance(rh, WeightedMap):
        raise ValueError("rh is not a WeightedMap")
    if not lh.norm or not rh.norm:
        return 1
    return dot_product(lh, rh) / sqrt(lh.norm * rh.norm)


def mutual_information_distance(lh, rh):
    return self_information(lh) + self_information(rh) \
        - 0.5 * self_information(merge(lh, rh))

def euclidean_similarity(lh, rh):
    lh_norm = lh.norm or 1
    rh_norm = rh.norm or 1
    sq  = lambda x: x * x
    key_set = set(chain(lh.keys(), rh.keys()))
    # l_norm  = len(lh) / len(key_set)
    # r_norm  = len(rh) / len(key_set)
    return 1 - sqrt(sum([sq(abs(lh[k]/lh_norm-rh[k]/rh_norm)) for k in key_set]))

default_similarity = cosine_similarity
