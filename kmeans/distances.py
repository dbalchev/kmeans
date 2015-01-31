from .utils import count_duplicates


def euclidean_distance(lh, rh):
    """
    Прави нормализирано евклидово разстояние на lh и rh;
    lh и rh трябва да са сортиран вектор от индекси на думи
    """
    if len(lh) == 0 or len(rh) == 0:
        return 0
    li = 0
    ri = 0
    nominator = 0
    while li < len(lh) or ri < len(rh):
        if li < len(lh) and (ri >= len(rh) or lh[li] < rh[ri]):
            cnt = count_duplicates(lh, li)
            nominator += cnt * cnt
            li += cnt
            continue
        if ri < len(rh) and (li >= len(lh) or lh[li] > rh[ri]):
            cnt = count_duplicates(rh, ri)
            nominator += cnt * cnt
            ri += cnt
            continue
        left_count = count_duplicates(lh, li)
        right_count = count_duplicates(rh, ri)
        delta = left_count - right_count
        nominator += delta * delta
        li += left_count
        ri += right_count

    denominator = len(lh) + len(rh)
    return nominator / denominator