import threading 
from multiprocess import Process
from entrenar_modelos import entrenar_modelo_rfc, entrenar_modelo_knn, entrenar_modelo_ann

funciones_entrenar = {
    "RFC": entrenar_modelo_rfc,
    "KNN": entrenar_modelo_knn,
    "ANN": entrenar_modelo_ann
}

def single(X_scaled, y, kfold, metodo):
    """
    Realiza el entrenamiento del modelo de manera secuencial.
    Según el valor metodo llamará a la función de entrenamiento correspondiente.
    Args:
        X_scaled (np.ndarray): Datos de entrada escalados.
        y (np.ndarray): Etiquetas de los datos.
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        metodo (str): Método de entrenamiento a utilizar ("RFC", "KNN", "ANN").
    """
    funcion_entrenar = funciones_entrenar[metodo]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        funcion_entrenar(X_scaled, train_idx, val_idx, y, "single")


def multihilo(X_scaled, y, kfold, metodo):
    """
    Entrena un modelo de clasificación secuencialmente.
    Según el valor metodo llamará a la función de entrenamiento correspondiente.
    Args:
        X_scaled (np.ndarray): Datos de entrada escalados.
        y (np.ndarray): Etiquetas de los datos.
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        metodo (str): Método de entrenamiento a utilizar ("RFC", "KNN", "ANN").
    """
    funcion_entrenar = funciones_entrenar[metodo]
    threads = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        thread = threading.Thread(target=funcion_entrenar, args=(X_scaled, train_idx, val_idx, y, "multihilo"))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join() 


def multiproceso(X_scaled, y, kfold, metodo):
    """
    Procede a realizar un entrenamiento secuencial del modelo.
    Según el valor metodo llamará a la función de entrenamiento correspondiente.

    Args:
        X_scaled (np.ndarray): Datos de entrada escalados.
        y (np.ndarray): Etiquetas de los datos.
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        metodo (str): Método de entrenamiento a utilizar ("RFC", "KNN", "ANN").
    """
    funcion_entrenar = funciones_entrenar[metodo]
    procesos = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        proceso = Process(target=funcion_entrenar, args=(X_scaled, train_idx, val_idx, y, "multiproceso"))
        proceso.start()
        procesos.append(proceso)
    for thread in procesos:
        thread.join() 

def n_jobs(X_scaled, y, kfold, metodo):
    """
    Entrena un modelo de clasificación utilizando múltiples trabajos en paralelo.
    Solo se utiliza para el método "RFC" o "KNN".

    Args:
        X_scaled (np.ndarray): Datos de entrada escalados.
        y (np.ndarray): Etiquetas de los datos.
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        metodo (str): Método de entrenamiento a utilizar ("RFC", "KNN").
    """
    funcion_entrenar = funciones_entrenar[metodo]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        funcion_entrenar(X_scaled, train_idx, val_idx, y, "n_jobs")
