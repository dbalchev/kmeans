import unittest

from .utils import count_duplicates
from .distances import euclidean_distance, dot_product

class EuclideanDistanceTest(unittest.TestCase):
    def test_count_dup(self):
        self.assertEqual(count_duplicates([1], 0), 1, "singleton")
        self.assertEqual(count_duplicates([1, 2], 0), 1, "different elements")
        self.assertEqual(count_duplicates([1, 1, 2], 0), 2, "result 2; different elements")
        self.assertEqual(count_duplicates([1, 1, 1], 0), 3, "all same")
        self.assertEqual(count_duplicates([1, 1, 1], 1), 2, "all same indx != 0")

    def test_dot(self):
        tests = [
            ([], [1,2,3], 0, "Empty sequence"),
            ([1, 2, 3], [1, 2, 3], 3, "Same sequence"),
            ([1, 1, 1], [1, 1, 1], 9, "Same sequnece repeating elements"),
            ([1], [2], 0, "Nonintersecting sequences"),
            ([1, 3, 5], [2, 4, 6], 0, "Nonintersecting longer sequences"),
            ([1, 2], [1, 3], 1, ""),
            ([1,1,1,1], [2,2,2], 0, "")
        ]
        for lh, rh, result, message in tests:
            r1 = dot_product(lh, rh)
            r2 = dot_product(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)

    def test_euclidean_distance(self):
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
            r1 = euclidean_distance(lh, rh)
            r2 = euclidean_distance(rh, lh)
            message = message or "{}, {}".format(lh, rh)
            self.assertEqual(r1, r2, message)
            self.assertEqual(r1, result, message)


if __name__ == "__main__":
    unittest.main()
