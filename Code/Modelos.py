# Librería NumPy
import numpy as np
# Librería Datetime
from datetime import datetime
# Librería Scikit-Learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from guardar_metricas import guardar_metricas
#Libreria para cronometras cuanto tiempo se tarda en ejecutar cada función
from timeit import timeit
# Script donde se encuentran los métodos de ejecución
from Seleccion_ejecucion import single, multihilo, multiproceso,n_jobs
#Script para entrenar el modelo
from entrenar_modelos import entrenar_modelo_rfc,entrenar_modelo_knn, entrenar_modelo_ann

def calculo_tiempos(k, X_scaled, y,n_componentes,reductor,modelo):
    """
    Realiza validación cruzada estratificada con un modelo XGBoost y calcula métricas de rendimiento.

    Args:
        k (int): Número de particiones para la validación cruzada.
        X_scaled(Dataframe): Lista de características extraídas de las imágenes.
        y (series): Lista con las etiquetas de las imágenes.
        n_componentes (int): Número de componentes principales a utilizar.
        reductor (str): Método de reducción de dimensionalidad ('PCA', 'ICA' o ninguno).
        modelo (str): Tipo de modelo a utilizar ('rfc' o 'knn').
    Returns:
        K_metrics (dict): Diccionario con las métricas de rendimiento.
    """
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    tiempo_secuencial=0
    tiempo_multihilo=0
    tiempo_multiproceso=0
    if reductor == 'PCA':
        pca = PCA(n_components=n_componentes, random_state=0)
        X_scaled = pca.fit_transform(X_scaled)
    elif reductor == 'ICA':
        ica = FastICA(n_components=n_componentes, random_state=0)
        X_scaled = ica.fit_transform(X_scaled)

    tiempo_secuencial = timeit(lambda: single(X_scaled, y, kfold,modelo), number=1)
    tiempo_multihilo = timeit(lambda: multihilo(X_scaled, y, kfold,modelo), number=1)
    tiempo_multiproceso = timeit(lambda: multiproceso(X_scaled, y, kfold,modelo), number=1)
    if modelo=='RFC' or modelo=='KNN':
        tiempo_n_jobs = timeit(lambda: n_jobs(X_scaled, y, kfold,modelo), number=1)
        k_metrics=metricas_rfc_knn(kfold,X_scaled, y, tiempo_secuencial, tiempo_multihilo, tiempo_multiproceso, tiempo_n_jobs,modelo)
    elif modelo=='ANN':
        k_metrics=metricas_ann(kfold,X_scaled, y, tiempo_secuencial, tiempo_multihilo, tiempo_multiproceso)
    return k_metrics



def metricas_rfc_knn(kfold,X_scaled, y, tiempo_secuencial, tiempo_multihilo, tiempo_multiproceso, tiempo_n_jobs, metodo):
    """
    Realiza validación cruzada estratificada con un modelo RFC o KNN y calcula métricas de rendimiento.
    
    Args:
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        X_scaled (DataFrame): Matriz de datos escalados que serán la entrada al modelo.
        y (Series): Vector de las variables de salida del modelo.
        tiempo_secuencial (float): Tiempo de ejecución secuencial.
        tiempo_multihilo (float): Tiempo de ejecución multihilo.
        tiempo_multiproceso (float): Tiempo de ejecución multiproceso.
        tiempo_n_jobs (float): Tiempo de ejecución con n_jobs.
        metodo (str): Método a utilizar ('RFC' o 'KNN').
    Returns:
        K_metrics (dict): Diccionario con las métricas de rendimiento.
    """
    funciones_entrenar = {
    "RFC": entrenar_modelo_rfc,
    "KNN": entrenar_modelo_knn
    }
    funcion_entrenar = funciones_entrenar[metodo]
    best_accuracy = 0
    k_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": [],"fpr_micro": [], "tpr_micro": [], "auc": [], "y_pred": [], "y_val": [], "y_val_bin": [], "y_score": [], "fecha_hora": [], "tiempo_secuencial": [], "tiempo_multihilo": [], "tiempo_multiproceso": [],"tiempo_n_jobs": []}
    best_combination = None
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        best_accuracy = 0
        best_combination = None
        modelo_fold, x_val, y_val=funcion_entrenar(X_scaled, train_idx, val_idx, y,"n_jobs")
        y_pred = modelo_fold.predict(x_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        if report['accuracy'] > best_accuracy:
            best_accuracy = report['accuracy']
            n_classes =  len(np.unique(y_val))                                  
            y_val_bin = label_binarize(y_val, classes=np.arange(0,n_classes,1)) 
            y_score = modelo_fold.predict_proba(x_val)
            fpr_micro, tpr_micro, _ = roc_curve(y_val_bin.ravel(), y_score[:,1].ravel())
            # Redondear los valores a 4 decimales
            fpr_micro = np.round(fpr_micro, 2)
            tpr_micro = np.round(tpr_micro, 2)
            auc_score = auc(fpr_micro, tpr_micro)
            best_combination = {
                'accuracy': report['accuracy'],
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score'],
                'fpr_micro': fpr_micro.tolist(),  
                'tpr_micro': tpr_micro.tolist(),  
                'auc': auc_score,
                'y_pred': y_pred.tolist(),
                'y_val': y_val.tolist(),
                'y_val_bin': y_val_bin.tolist(),
                'y_score': y_score.tolist(),
            }
        k_metrics["accuracy"].append(best_combination['accuracy'])
        k_metrics["precision"].append(best_combination['precision'])
        k_metrics["recall"].append(best_combination['recall'])
        k_metrics["f1_score"].append(best_combination['f1_score'])
        k_metrics["fpr_micro"].append(best_combination['fpr_micro'])
        k_metrics["tpr_micro"].append(best_combination['tpr_micro'])
        k_metrics["auc"].append(best_combination['auc'])
        k_metrics["y_pred"].append(best_combination['y_pred'])
        k_metrics["y_val"].append(best_combination['y_val'])
        k_metrics["y_val_bin"].append(best_combination['y_val_bin'])
        k_metrics["y_score"].append(best_combination['y_score'])
        k_metrics["fecha_hora"].append(datetime.now().strftime("%d-%m-%Y %H:%M"))
        print(f"Report de cada fold {fold+1}:\n", report)
    k_metrics["tiempo_secuencial"] = tiempo_secuencial
    k_metrics["tiempo_multihilo"] = tiempo_multihilo
    k_metrics["tiempo_multiproceso"] = tiempo_multiproceso
    k_metrics["tiempo_n_jobs"] = tiempo_n_jobs
    return k_metrics


def metricas_ann(kfold,X_scaled, y, tiempo_secuencial, tiempo_multihilo, tiempo_multiproceso):
    """
    Realiza validación cruzada estratificada con un modelo ANN y calcula métricas de rendimiento.

    Args:
        kfold (StratifiedKFold): Objeto de validación cruzada estratificada.
        X_scaled (DataFrame): Matriz de datos escalados que serán la entrada al modelo.
        y (Series): Vector de las variables de salida del modelo.
        tiempo_secuencial (float): Tiempo de ejecución secuencial.
        tiempo_multihilo (float): Tiempo de ejecución multihilo.
        tiempo_multiproceso (float): Tiempo de ejecución multiproceso.
    Returns:
        K_metrics (dict): Diccionario con las métricas de rendimiento.
    """
    k = 5
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=10)
    k_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": [],"fpr_micro": [], "tpr_micro": [], "auc": [], "y_pred": [], "y_val": [], "y_val_bin": [], "y_score": [], "fecha_hora": [], "train_accuracy": [], "train_loss": [], "val_accuracy": [], "val_loss": [],"tiempo_secuencial": [], "tiempo_multihilo": [], "tiempo_multiproceso": []}
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
        modelo_fold, x_val, y_val,history = entrenar_modelo_ann(X_scaled, train_idx, val_idx, y,"secuencial")
        n_classes =  len(np.unique(y_val))                                  
        y_val_bin = label_binarize(y_val, classes=np.arange(0,n_classes,1)) 
        y_score = modelo_fold.predict(x_val)
        fpr_micro, tpr_micro, _ = roc_curve(y_val_bin.ravel(), y_score[:,1].ravel())
        # Redondear los valores a 4 decimales
        fpr_micro = np.round(fpr_micro, 2)
        tpr_micro = np.round(tpr_micro, 2)
        auc_score = auc(fpr_micro, tpr_micro)
        y_pred = np.argmax(modelo_fold.predict(x_val), axis=1)
        report = classification_report(y_val, y_pred, output_dict=True)
        k_metrics["accuracy"].append(report['accuracy'])
        k_metrics["precision"].append(report['macro avg']['precision'])
        k_metrics["recall"].append(report['macro avg']['recall'])
        k_metrics["f1_score"].append(report['macro avg']['f1-score'])
        k_metrics["fpr_micro"].append(fpr_micro.tolist())
        k_metrics["tpr_micro"].append(tpr_micro.tolist())
        k_metrics["auc"].append(auc_score)
        k_metrics["y_pred"].append(y_pred.tolist())
        k_metrics["y_val"].append(y_val.tolist())
        k_metrics["y_val_bin"].append(y_val_bin.tolist())
        k_metrics["y_score"].append(y_score.tolist())
        k_metrics["fecha_hora"].append(datetime.now().strftime("%d-%m-%Y %H:%M"))
        k_metrics["train_accuracy"].append(history.history['accuracy'])
        k_metrics["train_loss"].append(history.history['loss'])
        k_metrics["val_accuracy"].append(history.history['val_accuracy'])
        k_metrics["val_loss"].append(history.history['val_loss'])
        k_metrics["tiempo_secuencial"].append(tiempo_secuencial)
        k_metrics["tiempo_multihilo"].append(tiempo_multihilo)
        k_metrics["tiempo_multiproceso"].append(tiempo_multiproceso)
    return k_metrics




def model(X_scaled,y,n_componentes,reductor,modelo):
    """
    Entrena y evalúa un modelo XGBoost utilizando características escaladas.

    Args:
        X_scaled(Dataframe): Matriz de datos escalados que seran la entrada al modelo.
        y(Series): Vector de las variables de salida del modelo.
    """
    k = 5
    k_metrics = calculo_tiempos(k, X_scaled, y,n_componentes,reductor,modelo)
    if modelo=='RFC' or modelo=='KNN':
        guardar_metricas(k, k_metrics, 'Resultados_'+modelo+'_'+reductor+'_'+str(n_componentes)+'.xlsx', histories=False, roc=True,n_jobs=True)
    elif modelo=='ANN':
        guardar_metricas(k, k_metrics, 'Resultados_'+modelo+'_'+reductor+'_'+str(n_componentes)+'.xlsx', histories=True, roc=True,n_jobs=False)
    