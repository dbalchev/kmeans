from pathlib import Path
from collections import Counter
from itertools import chain

def _process_line(line, corpus_root):
    filename, *categories = line.split()
    return corpus_root.joinpath(filename), categories

class Categories:
    def __init__(self, category_filename, corpus_prefix="."):
        self.corpus_root = Path(corpus_prefix)
        corpus_root = self.corpus_root
        with open(category_filename) as inp:
            self.file_categories = dict(_process_line(line, corpus_root) \
                for line in inp)

    def __getitem__(self, filename):
        return self.file_categories.get(Path(filename), [])

    def most_common(self, filename_iterable, max_results = None):
        categories_iter = chain.from_iterable(self[filename] \
            for filename in filename_iterable)
        return Counter(categories_iter).most_common(max_results)
