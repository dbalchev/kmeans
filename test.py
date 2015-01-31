from kmeans.utils import Vectorizer, WeightedMap
from kmeans.distances import cosine_distance, mutual_information_distance
# import kmeans

vectorizer = Vectorizer()
file1, file2 = "reuters/training/1", "reuters/training/5"
v1, v2 = (list(vectorizer.vectorize_file(f)) for f in (file1, file2))
v1, v2 = (WeightedMap(sorted(v)) for v in (v1, v2))

# print(v1)
# print(v2)


# print(cosine_distance(v1, v2))
# print(mutual_information_distance(v1, v1))
# print(mutual_information_distance(v1, v2))
# print(mutual_information_distance(v1, WeightedMap([1,2,3])))
# print(mutual_information_distance(WeightedMap([1,2,3]), WeightedMap([1,2,3])))

# Testing kmeans
from glob import iglob
from itertools import islice
from kmeans import clusterize

files = list(islice(iglob("./reuters/training/*"), 15))
clusters = clusterize(3, files)
print('======================')
for cluster_name, cluster in clusters.items():
	print('Cluster Name:',cluster_name)
	for item in cluster.items:
		print('\t',item.name)

