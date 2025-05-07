from leer_metricas import read_excel_file, get_data
import matplotlib.pyplot as plt
import statistics
from scipy.stats import kruskal
import numpy as np
from statsmodels.sandbox.stats.multicomp import MultiComparison
import os

def diagrama_cajas(accuracy):
    """
    Genera un diagrama de cajas (boxplot) para comparar la precisión de diferentes modelos.

    :param dict accuracy: Diccionario que contiene las precisiones de los modelos.
    :return: None
    """
    data = [accuracy['Param_default'], accuracy['ICA_4'], accuracy['ICA_8'], accuracy['ICA_11'], accuracy['PCA_4'], accuracy['PCA_8'], accuracy['PCA_11']]
    _, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Extracción de características')
    ax.boxplot(data, labels=[f'Default', 'ICA features=4',  'ICA features=8', 'ICA features=11', 'PCA features=4', 'PCA features=8', 'PCA features=11'])
    plt.show()



def comparar_todos_modelos(acc,modelo):
    """
    Compara todos los modelos usando la prueba de Kruskal-Wallis y la prueba de Tukey si hay diferencias significativas.

    :param dict acc: Diccionario con las precisiones de los modelos.
    :return: El nombre del mejor modelo y su media de precisión.
    :rtype: tuple(str, float)
    """
    alpha = 0.01
    F_statistic, pVal = kruskal(acc['Param_default'], acc['ICA_4'], acc['ICA_8'], acc['ICA_11'], acc['PCA_4'], acc['PCA_8'], acc['PCA_11'])

    with open('resultados_'+modelo+'.txt', 'a') as f:
        f.write(f'\n Comparación entre todos los modelos \n \n ')
        f.write(f'p-valor KrusW: {pVal}\n')
        if pVal <= alpha: 
            f.write('Rechazamos la hipótesis: los modelos son diferentes\n')
            stacked_data = np.hstack((acc['Param_default'], acc['ICA_4'], acc['ICA_8'], acc['ICA_11'], acc['PCA_4'], acc['PCA_8'], acc['PCA_11']))
            stacked_model = np.hstack((np.repeat('Default', len(acc['Param_default'])),
                                       np.repeat('ICA features=4', len(acc['ICA_4'])),
                                       np.repeat('ICA features=8', len(acc['ICA_8'])),
                                       np.repeat('ICA features=11', len(acc['ICA_11'])),
                                       np.repeat('PCA features=4', len(acc['PCA_4'])),
                                       np.repeat('PCA features=8', len(acc['PCA_8'])),
                                       np.repeat('PCA features=11', len(acc['PCA_11']))))

            MultiComp = MultiComparison(stacked_data, stacked_model)
            resultado_tukey = MultiComp.tukeyhsd(alpha=0.05)
            f.write(str(resultado_tukey) + '\n')

            medias = {
                'Default': statistics.mean(acc['Param_default']),
                'ICA features=4': statistics.mean(acc['ICA_4']),
                'ICA features=8': statistics.mean(acc['ICA_8']),
                'ICA features=11': statistics.mean(acc['ICA_11']),
                'PCA features=4': statistics.mean(acc['PCA_4']),
                'PCA features=8': statistics.mean(acc['PCA_8']),
                'PCA features=11': statistics.mean(acc['PCA_11'])
            }
            mejor_modelo = max(medias, key=medias.get)
            mejor_media = medias[mejor_modelo]
            # Verificar si el mejor modelo es estadísticamente diferente de los demás
            modelos_similares = []
            for i in range(len(resultado_tukey._results_table.data[1:])):
                grupo1, grupo2, _, _, _, _, reject = resultado_tukey._results_table.data[1:][i]
            # Si el mejor modelo está en la comparación y no hay diferencia significativa (reject == False)
            if (grupo1 == mejor_modelo or grupo2 == mejor_modelo) and not reject:
                modelos_similares.append(grupo1 if grupo1 != mejor_modelo else grupo2)
            if modelos_similares:
                f.write(f'Mejor modelo: {mejor_modelo} con Accuracy media: {float(mejor_media):.2f}, PERO es similar a: {", ".join(modelos_similares)}\n')
            else:
                f.write(f'Mejor modelo: {mejor_modelo} con Accuracy media: {float(mejor_media):.2f}, y es es diferente al resto de modelos.\n')
            return mejor_modelo, mejor_media
        else:
            f.write('Aceptamos la hipótesis: los modelos son iguales\n')
            return 'PCA features=4 ==>', statistics.mean(acc['PCA_4'])

def inicio_comp(Param_default,ICA_4,ICA_8,ICA_11,PCA_4,PCA_8,PCA_11,modelo):
    """
    Función principal que inicia el proceso de comparación de modelos.

    Lee los archivos de precisión, compara modelos y guarda los resultados.

    :return: None
    """
    if os.path.exists('resultados_'+modelo+'.txt'):
        os.remove('resultados_'+modelo+'.txt')  # Elimina el archivo
    accuracy ={'Param_default':[], 'ICA_4':[], 'ICA_8':[], 'ICA_11':[], 'PCA_4':[], 'PCA_8':[], 'PCA_11':[]}
    # Accuracy parametros por defecto
    data = read_excel_file(Param_default)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['Param_default']=k_metrics['accuracy']
    # Accuracy de HOG+SVC
    data = read_excel_file(ICA_4)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['ICA_4']=k_metrics['accuracy']
    # Accuracy de LBP+SVC
    data = read_excel_file(ICA_8)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['ICA_8']=k_metrics['accuracy']
    # Accuracy de HOG+XGBC
    data = read_excel_file(ICA_11)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['ICA_11']=k_metrics['accuracy']
    # Accuracy de LBP+XGBC
    data = read_excel_file(PCA_4)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['PCA_4']=k_metrics['accuracy']
    # Accuracy de HOG+RNN
    data = read_excel_file(PCA_8)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['PCA_8']=k_metrics['accuracy']
    # Accuracy de LBP+RNN
    data = read_excel_file(PCA_11)
    k_metrics,_, _, _, _ = get_data(data)
    accuracy['PCA_11']=k_metrics['accuracy']

    #diagrama_cajas(accuracy)
    diagrama_cajas(accuracy)
    modelo,media=comparar_todos_modelos(accuracy,modelo)
