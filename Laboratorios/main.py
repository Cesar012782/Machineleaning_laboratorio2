from Unsupervised.Clustering.kmeans import my_kmeans
from Unsupervised.Clustering.kmedoids import KMedoids

from Unsupervised.Utils.data_generation import generate_sample_data

X, y = generate_sample_data()
mykmeans = my_kmeans(X, y)