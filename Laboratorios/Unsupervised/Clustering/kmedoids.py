# *****************************
#     KMedoidsß Module
# *****************************
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
from unsupervised.utils.data_generation import generate_sample_data
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def my_kmedoids(X, n_clusters=8, max_iter=300):
    medoid_indices = np.random.choice(len(X), size=n_clusters, replace=False) # Paso 1: Inicializar los medoides de manera aleatoria
    medoids = X[medoid_indices]

def evaluate_kmedoids(X, max_clusters=5, max_iter=300):
    silhouette_scores = []
    db_scores = []

    for k in range(1, max_clusters + 1):
        medoids, labels = my_kmedoids(X, n_clusters=k, max_iter=max_iter)

        silhouette = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        silhouette_scores.append(silhouette)
        db_scores.append(db)

        print(f"Number of clusters: {k}, Silhouette Score: {silhouette}, Davies-Bouldin Score: {db}")

    plt.figure(figsize=(10, 5)) # Plot Silhouette Scores
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.subplot(1, 2, 2) # Plot Davies-Bouldin Scores
    plt.plot(range(1, max_clusters + 1), db_scores, marker='o')
    plt.title('Davies-Bouldin Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')

    plt.tight_layout()
    plt.show()