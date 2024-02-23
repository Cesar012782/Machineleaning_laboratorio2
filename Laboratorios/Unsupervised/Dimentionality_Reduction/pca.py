# *****************************
#     PCA Module
# *****************************

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def my_pca(X, n_components=2):
    # Paso 1: Escalar los datos
    scaling = StandardScaler()
    scaled_data = scaling.fit_transform(X)

    # Paso 2: Aplicar PCA
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    X_pca = pca.transform(scaled_data)

    # Paso 3: Visualizar la Varianza Acumulativa
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Acumulativa Explicada')
    plt.show()

    return X_pca