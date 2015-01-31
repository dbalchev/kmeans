from utils import Vectorizer
import unittest


def count_duplicates(lst, indx):
    cnt = 1
    el = lst[indx]
    indx += 1
    while indx < len(lst) and lst[indx] == el:
        indx += 1
        cnt += 1
    return cnt

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

class EuclideanDistanceTest(unittest.TestCase):
    def test_count_dup(self):
        self.assertEqual(count_duplicates([1], 0), 1, "singleton")
        self.assertEqual(count_duplicates([1, 2], 0), 1, "different elements")
        self.assertEqual(count_duplicates([1, 1, 2], 0), 2, "result 2; different elements")
        self.assertEqual(count_duplicates([1, 1, 1], 0), 3, "all same")
        self.assertEqual(count_duplicates([1, 1, 1], 1), 2, "all same indx != 0")

    def test_euclidean_distance(self):
        tests = [
            ([], [1,2,3], 0, "Empty sequence"),
            ([1, 2, 3], [1, 2, 3], 0, "Same sequence"),
            ([1, 1, 1], [1, 1, 1], 0, "Same sequnece repeating elements"),
            ([1], [2], 1, "Nonintersecting sequences"),
            ([1, 3, 5], [2, 4, 6], 1, "Nonintersecting longer sequences"),
            ([1, 2], [1, 3], 0.5, ""),
        ]
        for lh, rh, result, message in tests:
            r1 = euclidean_distance(lh, rh)
            r2 = euclidean_distance(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)


def kmeans(n_clusters, text_seq, similarity=None):
    """
    Функция използваща k-means алогоритъм за клъстеризация на текстове
    text_seq трябва да са двойки (label, text_vector), където text_vector е
    вектор върнат от Vectorizer
    similarity = similarity or default_similarity
    """
    corpus = [(label, list(sorted(text))) for label, text in text_seq]
    raise Exception("finish me")

def clusterize(n_clusters, filename_seq, similarity=None):
    vec = Vectorizer()
    return kmeans(n_clusters, ((filename, vec.vectorize_file(filename)) \
        for filename in filename_seq))

if __name__ == "__main__":
    unittest.main()
