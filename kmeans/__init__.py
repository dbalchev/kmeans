from .utils import Vectorizer


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
