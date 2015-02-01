from kmeans.distances import cosine_similarity, mutual_information_distance, euclidean_similarity
from kmeans.utils import WeightedMap
from kmeans.vectorizers import Vectorizer
# import kmeans

vectorizer = Vectorizer()
file1, file2 = "reuters/training/1", "reuters/training/5"
v1, v2 = (list(vectorizer.vectorize_file(f)) for f in (file1, file2))
v1, v2 = (WeightedMap.from_vec(sorted(v)) for v in (v1, v2))

# print(v1)
# print(v2)


# print(cosine_distance(v1, v2))
# print(mutual_information_distance(v1, v1))
# print(mutual_information_distance(v1, v2))
# print(mutual_information_distance(v1, WeightedMap.from_vec([1,2,3])))
# print(mutual_information_distance(WeightedMap.from_vec([1,2,3]), WeightedMap.from_vec([1,2,3])))

# Testing kmeans
from glob import iglob
from itertools import islice, chain
from kmeans import KMeans

files = list(islice(iglob("./reuters/training/*"), 1000))
kmeans = KMeans(n_clusters=10, filename_seq=files,
				iterations_callback=lambda x: print("Done in", x, "iterations."))

clusters = kmeans.clusterize()
print('======================')
for cluster_name, cluster in clusters.items():
	print(cluster_name, "has", len(cluster.items), "docs")
	# for item in cluster.items:
	# 	print('\t',item.name)
#
# items = list(chain.from_iterable(cluster.items for cluster in clusters.values()))
# import csv
# with open("results.csv", "w") as csv_file:
#     writer = csv.DictWriter(csv_file, ["name"] + [item.name for item in items])
#     writer.writeheader()
#     for cu_item in items:
#         d = {item.name:cosine_similarity(item.content, cu_item.content) for item in items}
#         d["name"] = cu_item.name
#         writer.writerow(d)

kmeans.similarity_method = euclidean_similarity
clusters = kmeans.clusterize()
print('======================')
for cluster_name, cluster in clusters.items():
	print(cluster_name, "has", len(cluster.items), "docs")
