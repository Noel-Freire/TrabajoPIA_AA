# Librería Pandas
import pandas as pd

def guardar_metricas(k, metrics, file, histories, roc,n_jobs):
    """
    Guarda las métricas del modelo en un archivo Excel, incluyendo información sobre las predicciones, 
    evolución del entrenamiento y curvas ROC si están disponibles.

    El archivo Excel generado puede contener varias hojas:
    - `k_metrics`: Métricas generales del modelo por cada fold de la validación cruzada.
    - `y_test`: Clases reales de las imágenes utilizadas en la validación.
    - `y_pred`: Clases predichas por el modelo.
    - `histories` (opcional): Evolución del accuracy y la pérdida durante el entrenamiento.
    - `roc` (opcional): Datos para la curva ROC global y por clases.

    :param int k: Número de folds en la validación cruzada.
    :param dict metrics: Diccionario con las métricas del modelo, incluyendo las predicciones y opcionalmente 
                         datos de entrenamiento y curvas ROC.
    :param str file: Nombre del archivo Excel donde se guardarán las métricas.
    :param bool histories: Indica si `metrics` contiene los datos de evolución del accuracy y pérdida.
    :param bool roc: Indica si `metrics` contiene los datos para generar curvas ROC.
    :return: No retorna ningún valor, solo genera y guarda un archivo Excel con las métricas.
    :rtype: None
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df['fold'] = range(1, k+1)
    if n_jobs == True:
        columns = ['fecha_hora'] + ['fold'] + ['accuracy'] + ['precision'] + ['recall'] + ['f1_score'] + ["tiempo_secuencial"] + ["tiempo_multihilo"]+ ["tiempo_multiproceso"]+ ["tiempo_n_jobs"]
    else:
        columns = ['fecha_hora'] + ['fold'] + ['accuracy'] + ['precision'] + ['recall'] + ['f1_score'] + ["tiempo_secuencial"] + ["tiempo_multihilo"]+ ["tiempo_multiproceso"]
    if 'C' in metrics:
        columns = columns + ['C']
    if 'solver' in metrics:
        columns = columns + ['solver']
    if 'penalty' in metrics:
        columns = columns + ['penalty']
    if 'max_iter' in metrics:
        columns = columns + ['max_iter']
    k_metrics_df = metrics_df[columns]
    y_test_df = pd.DataFrame(metrics['y_val'])
    y_pred_df = pd.DataFrame(metrics['y_pred'])
    if histories == True:
        columns = ['train_loss'] + ['train_accuracy'] + ['val_loss'] + ['val_accuracy']
        histories_df = metrics_df[columns]
    if roc == True:
        columns = ['fpr_micro'] + ['tpr_micro'] + ['auc'] 
        roc_df = metrics_df[columns]
    if (histories == False and roc == False):
        with pd.ExcelWriter(file) as writer:
            k_metrics_df.to_excel(writer, sheet_name='k_metrics')
            y_test_df.to_excel(writer, sheet_name='y_test')
            y_pred_df.to_excel(writer, sheet_name='y_pred')
    elif (histories == True and roc == False):
        with pd.ExcelWriter(file) as writer:
            k_metrics_df.to_excel(writer, sheet_name='k_metrics')
            y_test_df.to_excel(writer, sheet_name='y_test')
            y_pred_df.to_excel(writer, sheet_name='y_pred')
            histories_df.to_excel(writer, sheet_name='histories')
    elif (histories == False and roc == True):
        with pd.ExcelWriter(file) as writer:
            k_metrics_df.to_excel(writer, sheet_name='k_metrics')
            y_test_df.to_excel(writer, sheet_name='y_test')
            y_pred_df.to_excel(writer, sheet_name='y_pred')
            roc_df.to_excel(writer, sheet_name='roc')
    else:
        with pd.ExcelWriter(file) as writer:
            k_metrics_df.to_excel(writer, sheet_name='k_metrics')
            y_test_df.to_excel(writer, sheet_name='y_test')
            y_pred_df.to_excel(writer, sheet_name='y_pred')
            histories_df.to_excel(writer, sheet_name='histories')
            roc_df.to_excel(writer, sheet_name='roc')