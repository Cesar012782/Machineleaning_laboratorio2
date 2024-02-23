# *****************************
#     KMedoidsß Module
# *****************************
import numpy as np
from sklearn_extra.cluster import KMedoids

def my_kmedoids(X, n_clusters=8, max_iter=300):
    # Paso 1: Inicializar los medoides de manera aleatoria
    medoid_indices = np.random.choice(len(X), size=n_clusters, replace=False)
    medoids = X[medoid_indices]

    # Paso 2: Calcular las distancias entre los puntos y los medoides
    distances = np.zeros((len(X), n_clusters))
    for i, medoid in enumerate(medoids):
        distances[:, i] = np.linalg.norm(X - medoid, axis=1)

    # Paso 3: Asignar cada punto al medoide más cercano
    labels = np.argmin(distances, axis=1)

    # Paso 4: Actualizar los medoides iterativamente
    for _ in range(max_iter):
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            cluster_distances = distances[cluster_indices, :]
            cluster_distances_sum = np.sum(cluster_distances, axis=0)
            medoid_index = np.argmin(cluster_distances_sum)
            medoids[i] = X[cluster_indices[medoid_index]]

            # Actualizar las distancias
            for j, medoid in enumerate(medoids):
                distances[:, j] = np.linalg.norm(X - medoid, axis=1)

            labels = np.argmin(distances, axis=1)

    return medoids, labels