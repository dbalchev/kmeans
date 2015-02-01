from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .stopwords import STOPWORDS

stemmer = WordNetLemmatizer()


def stem(string):
    """
    За да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> wordnet
    """
    return stemmer.lemmatize(string)


def tokenize(filename):
    """
    За да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> punkt
    """
    with open(filename) as inp:
        return (stem(token) for token in word_tokenize(inp.read(-1)))


class Prenum(dict):
    def __init__(self):
        super().__init__(self)
        self.__counter = 0

    def __missing__(self, key):
        new_num = self.__counter
        self.__counter += 1
        self[key] = new_num
        return new_num


class Vectorizer:
    def __init__(self):
        self.prenum = Prenum()
        self.stopwords = STOPWORDS

    def vectorize_seq(self, seq):
        return (self.prenum[token] for token in seq \
            if token.isalpha() and token not in self.stopwords)

    def vectorize_file(self, filename):
        return list(self.vectorize_seq(tokenize(filename)))
