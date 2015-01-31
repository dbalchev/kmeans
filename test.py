from kmeans.utils import Vectorizer, WeightedMap
from kmeans.distances import cosine_similarity, mutual_information_distance
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
from itertools import islice, chain
from kmeans import clusterize

files = list(islice(iglob("./reuters/training/*"), 50))
clusters = clusterize(5, files)
print('======================')
for cluster_name, cluster in clusters.items():
	print('Cluster Name:',cluster_name)
	for item in cluster.items:
		print('\t',item.name)

items = list(chain.from_iterable(cluster.items for cluster in clusters.values()))
import csv
with open("results.csv", "w") as csv_file:
    writer = csv.DictWriter(csv_file, ["name"] + [item.name for item in items])
    writer.writeheader()
    for cu_item in items:
        d = {item.name:cosine_similarity(item.content, cu_item.content) for item in items}
        d["name"] = cu_item.name
        writer.writerow(d)
