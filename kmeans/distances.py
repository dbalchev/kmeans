from .utils import count_duplicates, self_information, merge
from math import sqrt

def dot_product(lh, rh):
    if len(lh) > len(rh):
        lh, rh = rh, lh
    result = 0
    for l_word, l_weight in lh.items():
        result += l_weight * rh[l_word]
    return result

def cosine_distance(lh, rh):
    """
    Прави нормализирано евклидово разстояние на lh и rh;
    lh и rh трябва да са сортиран вектор от индекси на думи
    """

    if not lh.norm or not rh.norm:
        return 0
    return 1 - dot_product(lh, rh) / sqrt(lh.norm * rh.norm)

def mutual_information_distance(lh, rh):
    return self_information(lh) + self_information(rh) \
        - 0.5 * self_information(merge(lh, rh))

default_similarity = cosine_distance
