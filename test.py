from kmeans.utils import Vectorizer
# import kmeans

vectorizer = Vectorizer()
file1, file2 = "reuters/training/1", "reuters/training/5"
v1, v2 = (list(vectorizer.vectorize_file(f)) for f in (file1, file2))

print(v1)
print(v2)
