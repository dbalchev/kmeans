from .utils import count_duplicates
from math import sqrt

def get_norm_len(vec):
    cu = 0
    res = 0
    while cu < len(vec):
        cnt = count_duplicates(vec, cu)
        res += cnt * cnt
        cu += cnt
    return res

def dot_product(lh, rh):
    result = 0
    if len(lh) == 0 or len(rh) == 0:
        return 0
    li = 0
    ri = 0
    while li < len(lh) or ri < len(rh):
        if li < len(lh) and (ri >= len(rh) or lh[li] < rh[ri]):
            cnt = count_duplicates(lh, li)
            li += cnt
            continue
        if ri < len(rh) and (li >= len(lh) or lh[li] > rh[ri]):
            cnt = count_duplicates(rh, ri)
            ri += cnt
            continue
        left_count = count_duplicates(lh, li)
        right_count = count_duplicates(rh, ri)
        result += left_count * right_count
        li += left_count
        ri += right_count
    return result

def euclidean_distance(lh, rh):
    """
    Прави нормализирано евклидово разстояние на lh и rh;
    lh и rh трябва да са сортиран вектор от индекси на думи
    """

    denominator = sqrt(get_norm_len(lh) * get_norm_len(rh))
    if not denominator:
        return 0
    nominator = dot_product(lh, rh)
    return 1 - nominator / denominator