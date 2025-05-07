# Librería Scikit-Learn
from sklearn.manifold import TSNE
# Librería MatPlotLib
import matplotlib.pyplot as plt

def tsne(x, cluster_labels, metodo):
    """
    Representa el conjunto de datos en 2D aplicando la reducción de dimensionalidad con t-SNE.

    Args:
        x (ndarray): Datos originales de entrada.
        cluster_labels (array-like): Etiquetas de cada dato.
        metodo (str): Nombre del método de clustering utilizado (para el título del gráfico).

    Returns:
        None: Muestra un gráfico con la representación t-SNE de los clústers.
    """
    x_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(x)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=cluster_labels)
    handles, labels = scatter.legend_elements(num=None)
    plt.legend(handles=handles, labels=labels, title='Grupos')
    plt.title(f'Representación de los clústers con t-SNE ({metodo})')
    plt.show()