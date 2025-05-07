# Librería Pandas
import pandas as pd

def guardar_metricas_clustering(file, metrics, etiquetas_real, etiquetas_pred, centroides=None):
    """
    Guarda las métricas de evaluación, etiquetas reales, etiquetas predichas y, si están disponibles, 
    los centroides del clustering en un archivo Excel.

    Args:
        file (str): Ruta y nombre del archivo Excel donde se guardarán los resultados.
        metrics (dict): Diccionario con métricas de evaluación (V-Measure e Índice de Rand ajustado).
        etiquetas_real (array-like): Etiquetas reales de los datos.
        etiquetas_pred (array-like): Etiquetas asignadas por el modelo de clustering.
        centroides (ndarray, optional): Coordenadas de los centroides de los clústers (si aplica).

    Returns:
        None: Los datos se guardan en el archivo Excel especificado.
    """
    metrics_df = pd.DataFrame(metrics)
    y_real = pd.DataFrame(etiquetas_real)
    y_pred = pd.DataFrame(etiquetas_pred)
    with pd.ExcelWriter(file) as writer:
        metrics_df.to_excel(writer, sheet_name='metrics')
        y_real.to_excel(writer, sheet_name='y_real')
        y_pred.to_excel(writer, sheet_name='y_pred')
        if centroides is not None:
            centroides = pd.DataFrame(centroides)
            centroides.to_excel(writer, sheet_name='centroides')