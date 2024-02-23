# *****************************
#     t-SNE Module
# *****************************
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from unsupervised.utils.data_generation import generate_sample_data

def my_tsne(data, n_components=2, perplexity=30, learning_rate=200):
    # Paso 1: Escalar los datos
    scaling = StandardScaler()
    scaled_data = scaling.fit_transform(data)

    # Paso 2: Aplicar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    data_tsne = tsne.fit_transform(scaled_data)

    # Visualizar los resultados (puedes personalizar esto según tus necesidades)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.show()

    return data_tsne