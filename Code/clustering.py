# Librería NumPy
import numpy as np
# Librería Datetime
from datetime import datetime
# Librería Scikit-Learn
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import v_measure_score, adjusted_rand_score
# Librería MatPlotLib
import matplotlib.pyplot as plt
# Funciones del módulo 'metricas_clustering.py'
from metricas_clustering import guardar_metricas_clustering
# Funciones del módulo 'tsne.py'
from tsne import tsne

def kmeans(x, y):
    """
    Aplica el algoritmo k-Means para encontrar el número óptimo de clústers 
    utilizando las métricas de inercia y coeficiente de silueta. Guarda los resultados 
    y visualiza los clústers con t-SNE.

    Args:
        x (ndarray): Datos de entrada para el clustering.
        y (array-like): Etiquetas reales de los datos (para evaluación).

    Returns:
        None: Los resultados se guardan en un archivo Excel y se muestra una gráfica.
    """
    metrics = {"fecha_hora": [], "v_measure": [], "adjusted_rand_index": []}
    modelos = []
    inercias = []
    coefs_silueta = []
    k_rango = range(2, 11)  # Valores de k desde 2 hasta 10
    for k in k_rango:
        modelo = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(x)
        modelos.append(modelo)
        labels = modelo.labels_
        coef_silueta = silhouette_score(x, labels)
        inercias.append(modelo.inertia_)
        coefs_silueta.append(coef_silueta)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    plt.grid(True)
    ax1.plot(k_rango, inercias, color='blue', label='Inercia')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inercia', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(k_rango, coefs_silueta, color='red', label='Coef. Silueta')
    ax2.set_ylabel('Coeficiente de silueta', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title("Inercia y Coeficiente de silueta vs Nº de clústers")
    plt.show()
    mejor_k_inercia = np.argmin(inercias)
    mejor_k_coef_silueta = np.argmax(coefs_silueta)
    print(f'El mejor valor de inercia es {inercias[mejor_k_inercia]:.2f} y se consigue con k={mejor_k_inercia+2}')
    print(f'El mejor valor de coeficiente de silueta es {coefs_silueta[mejor_k_coef_silueta]:.3f} y se consigue con k={mejor_k_coef_silueta+2}')
    mejor_modelo = modelos[mejor_k_coef_silueta]
    etiquetas = mejor_modelo.labels_
    centroides = mejor_modelo.cluster_centers_
    metrics['fecha_hora'].append(datetime.now().strftime("%d-%m-%Y %H:%M"))
    metrics['v_measure'].append(v_measure_score(y, etiquetas))
    metrics['adjusted_rand_index'].append(adjusted_rand_score(y, etiquetas))
    guardar_metricas_clustering(file='Resultados_kmeans.xlsx', metrics=metrics, etiquetas_real=y, etiquetas_pred=etiquetas, centroides=centroides)
    tsne(x, etiquetas, 'k-medias (k=2)')

def dbscan(x, y):
    """
    Aplica el algoritmo DBSCAN, calcula métricas 
    de calidad del clustering y visualiza la distribución de distancias para 
    seleccionar un buen valor de epsilon.

    Args:
        x (ndarray): Datos de entrada para el clustering.
        y (array-like): Etiquetas reales de los datos (para evaluación).

    Returns:
        None: Los resultados se guardan en un archivo Excel y se muestra una gráfica.
    """
    metrics = {"fecha_hora": [], "epsilon": [], "v_measure": [], "adjusted_rand_index": []}
    neigh = NearestNeighbors(n_neighbors=2*x.shape[1]) # minPts = 2 * dimensión
    neigh.fit(x)
    k_distancias, indices = neigh.kneighbors(x)
    distancia_k = k_distancias[:, -1]
    distancia_k_ordenada = np.sort(distancia_k)
    plt.figure(figsize=(10, 5))
    plt.plot(distancia_k_ordenada)
    plt.grid(True)
    plt.title("Distancias ordenadas a los 42-vecinos más cercanos")
    plt.show()
    for i in range(20,30,1):
        modelo = DBSCAN(eps=i/10, min_samples=2*x.shape[1])
        etiquetas = modelo.fit_predict(x)
        print(f'Grupos detectados: {etiquetas[np.argmax(etiquetas)]+1}') # Se suma 1 porque al primer grupo se le asigna el valor 0
        print(f'Datos clasificados como ruido: {list(etiquetas).count(-1)}')
        metrics['fecha_hora'].append(datetime.now().strftime("%d-%m-%Y %H:%M"))
        metrics['epsilon'].append(i/10)
        metrics['v_measure'].append(v_measure_score(y, etiquetas))
        metrics['adjusted_rand_index'].append(adjusted_rand_score(y, etiquetas))
    guardar_metricas_clustering(file='Resultados_DBSCAN.xlsx', metrics=metrics, etiquetas_real=y, etiquetas_pred=etiquetas)
    tsne(x, etiquetas, 'DBSCAN (eps = 2.9)')