# *****************************
#     t-SNE Module
# *****************************
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def my_tsne(X, n_components=2, perplexity=30, learning_rate=200):
    # Paso 1: Escalar los datos
    scaling = StandardScaler()
    scaled_data = scaling.fit_transform(X)

    # Paso 2: Aplicar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    X_tsne = tsne.fit_transform(scaled_data)

    # Visualizar los resultados (puedes personalizar esto según tus necesidades)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.show()

    return X_tsne