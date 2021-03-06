import sys
sys.path.insert(0, '.')
from kmeans.distances import cosine_similarity, mutual_information_distance, euclidean_similarity
from kmeans.utils import WeightedMap
from kmeans.vectorizers import Vectorizer
from kmeans.categories import Categories
# import kmeans

# file1, file2 = "reuters/training/1", "reuters/training/5"
# vectorizer = Vectorizer()
# v1, v2 = (list(vectorizer.vectorize_file(f)) for f in (file1, file2))
# v1, v2 = (WeightedMap.from_vec(sorted(v)) for v in (v1, v2))

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
from kmeans import KMeans, TFIDFDataBase
from cProfile import Profile
import csv
import random

profiler = Profile()
categories = Categories("./reuters/cats.txt", "./reuters")
files = list(iglob("./reuters/training/*"))
initstate = random.getstate()
kmeans = KMeans(n_clusters=30, filename_seq=files,
                max_iterations=100,
				iterations_callback=lambda x: print("Done in", x, "iterations."))
#
clusters = kmeans.clusterize()
random.setstate(initstate)
print('======================')
# for cluster_name, cluster in clusters.items():
# 	print(cluster_name, "has", len(cluster.items), "docs")
# 	for item in cluster.items:
# 		print('\t',item.name)

with open("results.txt", "w") as outp:
    for cluster_name, cluster in clusters.items():
        outp.write(
            "{} has {} docs\n{}\n".format(
                cluster_name, len(cluster.items),
                categories.most_common(
                    (fr.name for fr in cluster.items),
                    5)))
exit()
profiler.enable()
tfidf = TFIDFDataBase(kmeans.corpus)
# indices = [
#     (0, 0),
#     (1, 0),
#     (0, 1),
#     (1, 2),
#     (2, 3),
#     (1, 3),
#     (1, 1)
# ]
# for i, j in indices:
#     ic, jc = (kmeans.corpus[x].content for x in (i, j))
#     print(tfidf.euclidean_similarity(ic, jc), cosine_similarity(ic, jc))
#
# exit()
try:
    print("clusterize")
    kmeans.similarity_method = tfidf.euclidean_similarity
    clusters = kmeans.clusterize()
    # for cluster_name, cluster in clusters.items():
    # 	print(cluster_name, "has", len(cluster.items), "docs")
    # 	for item in cluster.items:
    # 		print('\t',item.name)
    items = list(chain.from_iterable(cluster.items for cluster in clusters.values()))
    # with open("results.csv", "w") as csv_file:
    #     writer = csv.DictWriter(csv_file, ["name"] + [item.name for item in items])
    #     writer.writeheader()
    #     for cu_item in items:
    #         d = {item.name:cosine_similarity(item.content, cu_item.content) for item in items}
    #         d["name"] = cu_item.name
    #         writer.writerow(d)

except KeyboardInterrupt:
    print("interupted")
else:
    print('======================')
    for cluster_name, cluster in clusters.items():
        print(cluster_name, "has", len(cluster.items), "docs")
finally:
   profiler.disable()
   profiler.dump_stats("profile.txt")
   print("we have profile")

# profiler.print_stats()
