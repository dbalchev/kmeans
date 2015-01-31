from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .stopwords import STOPWORDS
stemmer = WordNetLemmatizer()


# за да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> wordnet
def stem(s):
    return stemmer.lemmatize(s)

# за да работи трябва да се рънне python -m nltk.downloader -d <<хубава директория>> punkt
def tokenize(filename):
    with open(filename) as inp:
        return (stem(token) for token in word_tokenize(inp.read(-1)))

def count_duplicates(lst, indx):
    cnt = 1
    el = lst[indx]
    indx += 1
    while indx < len(lst) and lst[indx] == el:
        indx += 1
        cnt += 1
    return cnt

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
        return self.vectorize_seq(tokenize(filename))
