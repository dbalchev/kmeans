from math import log


def count_duplicates(lst, indx):
    cnt = 1
    while indx + cnt < len(lst) and lst[indx + cnt] == lst[indx]:
        cnt += 1
    return cnt


def merge(*args):
    args = [x for x in args if len(x)]
    result = WeightedMap.from_vec([])
    if not args:
        return result
    for wm in args:
        for key, weight in wm.items():
            result[key] += weight / len(args)
    norm = 0
    for w in result.values():
        norm += w * w
    result.norm = norm
    return result


def self_information(wm):
    si = 0.0
    for p in wm.values():
        si -= p * log(p, 2)
    return si


class WeightedMap(dict):
    def __init__(self, tupple_iterator):
        self.norm = 0
        super().__init__(tupple_iterator)
        self.norm = 0
        for w in self.values():
            self.norm += w * w

    def __hash__(self):
        return hash(id(self).to_bytes(4, "little"))

    def __eq__(self, oth):
        if self is oth:
            return True
        return super().__eq__(oth)

    def __ne__(self, oth):
        return not self == oth


    @staticmethod
    def from_vec(vec):
        if len(vec) == 0:
            return WeightedMap(())
        def gen():
            cu = 0
            denominator = len(vec)
            while cu < len(vec):
                cnt = count_duplicates(vec, cu)
                p = cnt / denominator
                yield vec[cu], p
                cu += cnt
        return WeightedMap(gen())

    def __missing__(self, key):
        return 0.0
