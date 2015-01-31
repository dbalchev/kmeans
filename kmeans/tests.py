from math import sqrt
import unittest

from .distances import cosine_distance, dot_product, merge, self_information, euclidean_similarity
from .utils import count_duplicates, WeightedMap


class cosineDistanceTest(unittest.TestCase):
    def test_count_dup(self):
        self.assertEqual(count_duplicates([1], 0), 1, "singleton")
        self.assertEqual(count_duplicates([1, 2], 0), 1, "different elements")
        self.assertEqual(count_duplicates([1, 1, 2], 0), 2, "result 2; different elements")
        self.assertEqual(count_duplicates([1, 1, 1], 0), 3, "all same")
        self.assertEqual(count_duplicates([1, 1, 1], 1), 2, "all same indx != 0")

    def test_vector_to_map(self):
        tests = [
            ([], {}),
            ([1], {1:1.0}),
            ([1,1], {1:1.0}),
            ([1, 2], {1:0.5, 2:0.5}),
            ([1, 1, 1, 2], {1:0.75, 2:0.25}),
        ]
        for vec, m in tests:
            self.assertEqual(WeightedMap(vec), m)

    def test_dot(self):
        tests = [
            ([], [1,2,3], 0, "Empty sequence"),
            ([1, 2, 3], [1, 2, 3], 1/3, "Same sequence"),
            ([1, 1, 1], [1, 1, 1], 1, "Same sequnece repeating elements"),
            ([1], [2], 0, "Nonintersecting sequences"),
            ([1, 3, 5], [2, 4, 6], 0, "Nonintersecting longer sequences"),
            ([1, 2], [1, 3], 0.25, ""),
            ([1,1,1,1], [2,2,2], 0, "")
        ]
        for lh, rh, result, message in tests:
            lh, rh = map(WeightedMap, (lh, rh))
            r1 = dot_product(lh, rh)
            r2 = dot_product(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)

    def test_euclidean_similarity(self):
        tests = [
            ([], [1,2,3], 0, "Empty sequence"),
            ([1, 2, 3], [1, 2, 3], 1, "Same sequence"),
            ([1, 1, 1], [1, 1, 1], 1, "Same sequnece repeating elements"),
            ([1], [2], 0, "Nonintersecting sequences"),
            ([1, 3, 5], [2, 4, 6], 0, "Nonintersecting longer sequences"),
            ([1, 2], [1, 3], 0.5, ""),
            ([1,1,1,1], [2,2,2], 0, "")
        ]
        for lh, rh, result, message in tests:
            lh, rh = map(WeightedMap, (lh, rh))
            r1 = euclidean_similarity(lh, rh)
            r2 = euclidean_similarity(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)

    def test_cosine_distance(self):
        tests = [
            ([], [1,2,3], 0, "Empty sequence"),
            ([1, 2, 3], [1, 2, 3], 0, "Same sequence"),
            ([1, 1, 1], [1, 1, 1], 0, "Same sequnece repeating elements"),
            ([1], [2], 1, "Nonintersecting sequences"),
            ([1, 3, 5], [2, 4, 6], 1, "Nonintersecting longer sequences"),
            ([1, 2], [1, 3], 0.5, ""),
            ([1,1,1,1], [2,2,2], 1, "")
        ]
        for lh, rh, result, message in tests:
            lh, rh = map(WeightedMap, (lh, rh))
            r1 = cosine_distance(lh, rh)
            r2 = cosine_distance(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)

    def test_merge(self):
        tests = [
            ([], [], []),
            ([], [1], [1]),
            ([1,2,3], [4,5,6], [1,2,3,4,5,6]),
            ([1,3,5], [2,4,6], [1,2,3,4,5,6])
        ]
        for lh, rh, result in tests:
            lh, rh, result = map(WeightedMap, (lh, rh, result))
            r1 = merge(lh, rh)
            r2 = merge(rh, lh)
            message = "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)

    def test_self_information(self):
        tests = [
            ([], 0),
            ([1], 0),
            ([1, 1, 1], 0),
            ([1, 2], 1)
        ]
        for test, result in tests:
            test = WeightedMap(test)
            r = self_information(test)
            message = "{}".format(test)
            self.assertEqual(r, result, message)
