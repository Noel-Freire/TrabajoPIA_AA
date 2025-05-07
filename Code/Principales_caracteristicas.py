from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca(x_scaled):
    """
    Realiza PCA sobre los datos escalados y visualiza la varianza explicada por cada componente principal.
    
    Args:
    x_scaled (DataFrame): Datos escalados.
    
    Returns:
        x_pca (ndarray): Datos que se enviarán al modelo.
    """
    #Aplicar PCA sobre los datos
    n_components = x_scaled.shape[1]
    pca = PCA(n_components=int(n_components))
    x_pca = pca.fit_transform(x_scaled)
    # Visualización de la varianza explicada
    plt.figure(figsize=(30,15))
    plt.bar(range(1, int(n_components)+1), pca.explained_variance_ratio_*100, color='blue')
    plt.xlabel("Componentes principales")
    plt.ylabel("% de varianza explicada")
    plt.title("Varianza explicada por cada componente principal (PCA)")
    plt.xticks(range(1, int(n_components)+1))
    plt.show()
    return x_pca