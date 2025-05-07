from subir_datos import cargar_datos
from normalizar import normalizar_datos
from Principales_caracteristicas import pca
from Matriz_correlacion import Variables_correlacion
from Modelos import model
from leer_metricas import plot_metrics
from comparacion_n_covariables import inicio_comp
# Librería NumPy
import numpy as np
# Funciones del módulo 'subir_datos.py'
from subir_datos import cargar_datos
# Funciones del módulo 'normalizar.py'
from normalizar import normalizar_datos
# Funciones del módulo 'clustering.py'
from clustering import kmeans, dbscan




if __name__ == '__main__':
    # Lectura de los datos del dataset
    x, y = cargar_datos('dataset_diabetes.csv','Diabetes_binary')
    # Normalizado de los datos
    X_scaled, scalers = normalizar_datos(x)
    # Clustering con k-medias
    #kmeans(X_scaled, y)
    # Clustering con DBSCAN
    #dbscan(X_scaled, y)

    # PCA y análisis de correlación
    x_pca=pca(X_scaled)
    #Se saca el número analizando la gráfica anterior
    n_componentes= 8       
    Variables_correlacion(X_scaled, x_pca[:,:n_componentes],y)
    #model(X_scaled,y,n_componentes,'PCA','RFC')
    #model(X_scaled,y,n_componentes,'ICA','RFC')
    #model(X_scaled,y,n_componentes,'DEFAULT','RFC')
    #model(X_scaled,y,n_componentes,'PCA','KNN')
    #model(X_scaled,y,n_componentes,'ICA','KNN')
    #model(X_scaled,y,n_componentes,'DEFAULT','KNN')
    #model(X_scaled,y,n_componentes,'PCA','ANN')
    #model(X_scaled,y,n_componentes,'ICA','ANN')
    #model(X_scaled,y,n_componentes,'DEFAULT','ANN')

    n_componentes= 4 
    #model(X_scaled,y,n_componentes,'PCA','RFC')
    #model(X_scaled,y,n_componentes,'ICA','RFC')
    #model(X_scaled,y,n_componentes,'PCA','KNN')
    #model(X_scaled,y,n_componentes,'ICA','KNN')
    #model(X_scaled,y,n_componentes,'PCA','ANN')
    #model(X_scaled,y,n_componentes,'ICA','ANN')

    n_componentes= 11
    #model(X_scaled,y,n_componentes,'PCA','RFC')
    #model(X_scaled,y,n_componentes,'ICA','RFC')
    #model(X_scaled,y,n_componentes,'PCA','KNN')
    #model(X_scaled,y,n_componentes,'ICA','KNN')
    #model(X_scaled,y,n_componentes,'PCA','ANN')
    #model(X_scaled,y,n_componentes,'ICA','ANN')
    
    #plot_metrics('Resultados_RFC_DEFAULT.xlsx')
    #plot_metrics('Resultados_RFC_PCA_4.xlsx')
    #plot_metrics('Resultados_RFC_PCA_8.xlsx')
    #plot_metrics('Resultados_RFC_PCA_11.xlsx')

    #plot_metrics('Resultados_RFC_ICA_4.xlsx')
    #plot_metrics('Resultados_RFC_ICA_8.xlsx')
    #plot_metrics('Resultados_RFC_ICA_11.xlsx')

    #plot_metrics('Resultados_KNN_DEFAULT.xlsx') 
    #plot_metrics('Resultados_KNN_PCA_4.xlsx') 
    #plot_metrics('Resultados_KNN_PCA_8.xlsx') 
    #plot_metrics('Resultados_KNN_PCA_11.xlsx') 

    #plot_metrics('Resultados_KNN_ICA_4.xlsx') 
    #plot_metrics('Resultados_KNN_ICA_8.xlsx') 
    #plot_metrics('Resultados_KNN_ICA_11.xlsx') 

    
    
    #plot_metrics('Resultados_ANN_DEFAULT.xlsx')
    #plot_metrics('Resultados_ANN_PCA_4.xlsx')
    #plot_metrics('Resultados_ANN_PCA_8.xlsx')
    #plot_metrics('Resultados_ANN_PCA_11.xlsx')

    #plot_metrics('Resultados_ANN_ICA_4.xlsx')
    #plot_metrics('Resultados_ANN_ICA_8.xlsx')
    #plot_metrics('Resultados_ANN_ICA_11.xlsx')
    
    # Comparación de modelos
    #inicio_comp('Resultados_RFC_DEFAULT.xlsx','Resultados_RFC_ICA_4.xlsx','Resultados_RFC_ICA_8.xlsx','Resultados_RFC_ICA_11.xlsx','Resultados_RFC_PCA_4.xlsx','Resultados_RFC_PCA_8.xlsx','Resultados_RFC_PCA_11.xlsx','RFC')
    inicio_comp('Resultados_KNN_DEFAULT.xlsx','Resultados_KNN_ICA_4.xlsx','Resultados_KNN_ICA_8.xlsx','Resultados_KNN_ICA_11.xlsx','Resultados_KNN_PCA_4.xlsx','Resultados_KNN_PCA_8.xlsx','Resultados_KNN_PCA_11.xlsx','KNN')
    #inicio_comp('Resultados_ANN_DEFAULT.xlsx','Resultados_ANN_ICA_4.xlsx','Resultados_ANN_ICA_8.xlsx','Resultados_ANN_ICA_11.xlsx','Resultados_ANN_PCA_4.xlsx','Resultados_ANN_PCA_8.xlsx','Resultados_ANN_PCA_11.xlsx','ANN')