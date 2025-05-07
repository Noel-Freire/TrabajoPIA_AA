import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def Variables_correlacion(X, x_pca, y):
    """
    Función que calcula la matriz de correlación entre las variables originales y la variable de salida,

    y entre las variables transformadas por PCA y la variable de salida.
    Se muestran las matrices de correlación en dos gráficos diferentes.

    Args:
        X (Dataframe): Datos no normalizados.
        x_pca (ndarray): Matriz de características transformadas por PCA.
        y(Series): Vector de las variables de salida del modelo.

    """
    nvar=X.shape[1]
    print("Número de variables originales: ", nvar)
    plt.figure(figsize=(20, 10))
    corr_mat = np.corrcoef(np.c_[X, y].T)
    etiquetas = [str(int) for int in range(1,nvar+1)]
    etiquetas.append('salida')
    sns.heatmap(corr_mat, vmin=-1, vmax=1, annot=True, linewidths=1, cmap='BrBG', 
        xticklabels=etiquetas, yticklabels=etiquetas,fmt=".2f");
    plt.title("Matriz de correlación entre las variables originales")
    plt.show()

    nvar=x_pca.shape[1]
    plt.figure(figsize=(20, 10))
    corr_mat = np.corrcoef(np.c_[x_pca,y].T)
    etiquetas = [str(int) for int in range(1,nvar+1)]
    etiquetas.append('salida')
    sns.heatmap(corr_mat, vmin=-1, vmax=1, annot=True, linewidths=1, cmap='BrBG', 
        xticklabels=etiquetas, yticklabels=etiquetas,fmt=".2f");
    plt.title("Matriz de correlación entre las variables transformadas por PCA")
    plt.show()