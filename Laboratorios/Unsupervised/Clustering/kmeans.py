# *****************************
#     MyKmeans Class Module
# *****************************
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from unsupervised.utils.data_generation import generate_sample_data
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
import numpy as np

def plot_silhouette(X, labels):
    cluster_labels = np.unique(labels)
    n_clusters = len(cluster_labels)
    silhouette_vals = silhouette_samples(X, labels, metric='euclidean')

    y_lower, y_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[labels == c]
        c_silhouette_vals.sort()
        y_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_lower, y_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)

        yticks.append((y_lower + y_upper) / 2)
        y_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()

def my_kmeans(X):
    silhouette_scores = []
    calinski_scores = []

    for k in range(1, 6):
        kmeans = KMeans(n_clusters=k, random_state=1)
        labels = kmeans.fit_predict(X)

        # Calcular y mostrar el diagrama de silueta
        plot_silhouette(X, labels)

        # Calcular y mostrar el coeficiente de Calinski-Harabasz
        calinski_score = calinski_harabasz_score(X, labels)
        print(f'Número de clusters: {k}, Coeficiente de Calinski-Harabasz: {calinski_score}')

        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_score)

    # Mostrar gráficamente la evolución de los scores
    plt.plot(range(1, 6), silhouette_scores, label='Silhouette Score')
    plt.plot(range(1, 6), calinski_scores, label='Calinski-Harabasz Score')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Score')
    plt.legend()
    plt.show()