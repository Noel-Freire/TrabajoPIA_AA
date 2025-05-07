from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from tensorflow.keras import optimizers, callbacks, models, layers
import tensorflow as tf
import numpy as np

def entrenar_modelo_rfc( X_scaled, train_idx, val_idx, y,mode):
    """
    Función que entrena el modelo de Random Forest utilizando diferentes métodos (secuencial, multihilo, multiproceso).
    
    Args:
        X_scaled (array): Datos de entrada escalados.
        train_idx (array): Índices de entrenamiento.
        val_idx (array): Índices de validación.
        y (array): Etiquetas de clase.
        mode (str): Modo de entrenamiento ('secuencial', 'multihilo', 'multiproceso', 'n_jobs').
    Returns:
        modelo_fold (RandomForestClassifier): Modelo entrenado.
        x_val (array): Datos de validación.
        y_val (array): Etiquetas de validación.
    """
    print(f"Entrenando modelo RFC ",{mode},"...")
    parametros_RFC = param_RFC()
    x_train, x_val = np.array(X_scaled)[train_idx], np.array(X_scaled)[val_idx]
    y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
    modelo_fold = clone(RandomForestClassifier())
    if mode == "n_jobs":
        n_jobs_value = -1
    else:
        n_jobs_value = parametros_RFC['n_jobs'][0]
    modelo_fold.set_params(
        n_estimators=parametros_RFC['n_estimators'][0],
        max_depth=parametros_RFC['max_depth'][0],
        n_jobs=n_jobs_value,
        criterion=parametros_RFC['criterion'][0],
        max_features=parametros_RFC['max_features'][0],
        bootstrap=parametros_RFC['bootstrap'][0],
        random_state=parametros_RFC['random_state'][0]
    )
    modelo_fold.fit(x_train, y_train)
    return modelo_fold, x_val, y_val

def param_RFC():
    """
    Función que define los parámetros del modelo Random Forest.

    Returns:
        parametros_RFC (dict): Parámetros del modelo Random Forest.
    """
    parametros_RFC = {
        'n_estimators': [1000],          # Número de árboles en el bosque
        'max_depth': [5],                # Profundidad máxima de cada árbol
        'n_jobs': [1],                  # Usar todos los núcleos disponibles
        'criterion': ['gini'],           # Alternativa: 'entropy'
        'max_features': ['sqrt'],        # Cuántas características considerar al dividir
        'bootstrap': [True],             # Si usar muestreo con reemplazo
        'random_state': [0]              # Para reproducibilidad
    }
    return parametros_RFC


def entrenar_modelo_knn(X_scaled, train_idx, val_idx, y,mode):
    """
    Función que entrena el modelo K-Nearest Neighbors utilizando diferentes métodos (secuencial, multihilo, multiproceso).

    Args:
        X_scaled (array): Datos de entrada escalados.
        train_idx (array): Índices de entrenamiento.
        val_idx (array): Índices de validación.
        y (array): Etiquetas de clase.
        mode (str): Modo de entrenamiento ('secuencial', 'multihilo', 'multiproceso', 'n_jobs').
    Returns:
        modelo_fold (KNeighborsClassifier): Modelo entrenado.
        x_val (array): Datos de validación.
        y_val (array): Etiquetas de validación.
    """
    print(f"Entrenando modelo KNN",{mode},"...")
    parametros_KNN = param_KNN()
    x_train, x_val = np.array(X_scaled)[train_idx], np.array(X_scaled)[val_idx]
    y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
    modelo_fold = clone(KNeighborsClassifier())
    if mode == "n_jobs":
        n_jobs_value = -1
    else:
        n_jobs_value = parametros_KNN['n_jobs'][0]
    modelo_fold.set_params(
        n_neighbors=parametros_KNN['n_neighbors'][0],
        weights=parametros_KNN['weights'][0],
        algorithm=parametros_KNN['algorithm'][0],
        leaf_size=parametros_KNN['leaf_size'][0],
        n_jobs=n_jobs_value,
        metric=parametros_KNN['metric'][0],
        p=parametros_KNN['p'][0],
    )
    modelo_fold.fit(x_train, y_train)
    return modelo_fold, x_val, y_val

def param_KNN():
    """
    Función que define los parámetros del modelo KNN.
    
    Returns:
        parametros_KNN (dict): Parámetros del modelo KNN.
    """
    parametros_KNN = {
        'n_neighbors':[260],
        'weights':['distance'],
        'algorithm':['auto'],
        'leaf_size':[40],
        'metric':['minkowski'],
        'n_jobs':[1],
        'p':[2 ]            
    }
    return parametros_KNN

def entrenar_modelo_ann(X_scaled, train_idx, val_idx, y,mode):
    """
    Función que entrena el modelo ANN utilizando diferentes métodos (secuencial, multihilo, multiproceso).

    Args:
        X_scaled (array): Datos de entrada escalados.
        train_idx (array): Índices de entrenamiento.
        val_idx (array): Índices de validación.
        y (array): Etiquetas de clase.
        mode (str): Modo de entrenamiento ('secuencial', 'multihilo', 'multiproceso', 'n_jobs').

    Returns:
        model (Sequential): Modelo entrenado.
        x_val (array): Datos de validación.
        y_val (array): Etiquetas de validación.
        history (History): Historial del entrenamiento.
    """
    print(f"Entrenando modelo ANN",{mode},"...")
    x_train, x_val = np.array(X_scaled)[train_idx], np.array(X_scaled)[val_idx]
    y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
    model = define_model_ann(caracteristicas=x_train.shape[1],num_clases=2)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.002),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=100,
        validation_data=(x_val, y_val),
        verbose=0,
        callbacks=[earlystop_callback]
    )
    return model, x_val, y_val,history



def define_model_ann(caracteristicas, num_clases):
    """
    Define un modelo de red neuronal con arquitectura ANN para clasificación multiclase.

    Args:
        caracteristicas (int): Número de características de entrada.
        num_clases (int): Número de clases para la clasificación.
    Returns:
        model (Sequential): Modelo ANN compilado.
    """
    model = models.Sequential([
    layers.Input(shape=(caracteristicas,)),  # El tamaño de las características 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),  # Dropout con tasa del 30%
    layers.Dense(64, activation='relu'),  # Capa densa con 64 neuronas y función de activación ReLU
    layers.BatchNormalization(),  # Normalización de lotes
    layers.Dropout(0.3),  # Dropout con tasa del 30%
    layers.Dense(32, activation='relu'),  # Otra capa densa
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_clases, activation='softmax')  # Capa de salida con 2
    ])
    return model