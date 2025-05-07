# Librería Pandas
import pandas as pd
# Librería NumPy
import numpy as np
# Librería Scikit-Learn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc
# Librería Matplotlib
import matplotlib.pyplot as plt
# Librería AST (Abstract Syntax Tree)
import ast

def read_excel_file(excel_file):
    """
    Lee un archivo Excel y devuelve su contenido en un diccionario, donde cada clave 
    representa el nombre de una hoja y su valor correspondiente es un DataFrame con los datos de dicha hoja.

    :param str excel_file: Ruta del archivo Excel a leer.
    :return: Diccionario con los datos de cada hoja en formato DataFrame.
    :rtype: dict[str, pandas.DataFrame]
    """
    excel_data = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
    return excel_data

def get_data(excel_data):
    """
    Extrae las métricas almacenadas en el diccionario generado por `read_excel_file`, 
    separándolas en distintos DataFrames.

    :param dict[str, pandas.DataFrame] excel_data: Diccionario con los datos del modelo.
    :return: 
        - **k_metrics** (*pandas.DataFrame*): Métricas generales del modelo por fold.
        - **y_test** (*pandas.DataFrame*): Clases reales de cada imagen.
        - **y_pred** (*pandas.DataFrame*): Clases predichas por el modelo.
        - **histories** (*pandas.DataFrame*): Evolución del accuracy y pérdida (si está disponible).
        - **roc** (*pandas.DataFrame*): Datos para la curva ROC y AUC (si está disponible).
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
    """
    k_metrics = excel_data['k_metrics'].iloc[:,1:6]
    y_test = excel_data['y_test'].iloc[:,1:].dropna(axis=1)
    y_pred = excel_data['y_pred'].iloc[:,1:].dropna(axis=1)
    if 'histories' in excel_data:
        histories = excel_data['histories'].iloc[:,1:]
    else:
        histories = pd.DataFrame()
    if 'roc' in excel_data:
        roc = excel_data['roc'].iloc[:,1:]
    else:
        roc = pd.DataFrame()
    return k_metrics, y_test, y_pred, histories, roc

def plot_confusion_matrix(y_test, y_pred):
    """
    Genera y muestra una matriz de confusión para cada fold de validación.

    :param pandas.DataFrame y_test: DataFrame con las clases reales.
    :param pandas.DataFrame y_pred: DataFrame con las clases predichas.
    :return: No retorna ningún valor, solo muestra la matriz de confusión.
    :rtype: None
    """
    cm = []
    for i in range(0,y_test.shape[0]):
        cm_i = confusion_matrix(y_test.iloc[i, :], y_pred.iloc[i, :], labels=[0, 1])
        cm.append(cm_i)
    labels = ['Diabetes', 'Non-Diabetes'] 
    fig, axes = plt.subplots(nrows=(len(cm)//3+(len(cm)%3>0)), ncols=3)
    axes = axes.flatten()
    for i in range(len(cm)):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=labels)
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
        axes[i].set_title(f'k = {i+1}')
    for i in range(len(cm),len(axes)):
        fig.delaxes(axes[i])
    plt.subplots_adjust(left=0.06, right=0.96, top=0.97, bottom=0.06)
    plt.show()

def plot_accuracy_evolution(histories):
    """
    Muestra la evolución del accuracy en entrenamiento y validación a lo largo de las iteraciones.

    :param pandas.DataFrame histories: DataFrame con la evolución del accuracy y la pérdida 
        en entrenamiento y validación.
    :return: No retorna ningún valor, solo muestra la gráfica de evolución del accuracy.
    :rtype: None
    """
    k = histories.shape[0]
    fig, axes = plt.subplots(nrows=(k//3+(k%3>0)), ncols=3)
    axes = axes.flatten()
    for i in range(k):
        accuracy = eval(histories.iloc[i, 1])
        val_accuracy = eval(histories.iloc[i, 3])   
        epochs = range(1, len(accuracy) + 1)
        axes[i].plot(epochs, accuracy, label='Train accuracy', marker='s')
        axes[i].plot(epochs, val_accuracy, label='Validation accuracy', marker='o')
        axes[i].set_title(f'k = {i+1}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Accuracy')
        axes[i].legend()
        axes[i].grid(True)
    for i in range(histories.shape[0],len(axes)):
        fig.delaxes(axes[i])
    plt.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.07)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.show()

def plot_roc_curves(roc):
    """
    Genera y muestra la curva ROC global junto con el área bajo la curva (AUC).

    :param pandas.DataFrame roc: DataFrame con los datos para la curva ROC global.
    :return: No retorna ningún valor, solo muestra la curva ROC.
    :rtype: None
    """
    k = roc.shape[0]
    fig, axes = plt.subplots(nrows=(k//3+(k%3>0)), ncols=3)
    axes = axes.flatten()
    for i in range(k):
        fpr_micro = roc.iloc[i,0]
        tpr_micro = roc.iloc[i,1]
        fpr_micro = np.fromstring(roc.iloc[i, 0].strip("[]"), sep=",")
        tpr_micro = np.fromstring(roc.iloc[i, 1].strip("[]"), sep=",")
        roc_auc_micro = roc.iloc[i,2]
        axes[i].plot(fpr_micro, tpr_micro, color='red', lw=2, label='Curva ROC micro-average (AUC = %0.3f)' % roc_auc_micro)
        axes[i].plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
        axes[i].set_title(f'k = {i+1}')
        axes[i].set_xlabel('1 - Especificidad')
        axes[i].set_ylabel('Sensibilidad')
        axes[i].grid(True)
        axes[i].legend(loc="lower right")
    for i in range(roc.shape[0],len(axes)):
        fig.delaxes(axes[i])
    plt.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.07)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.show()

def plot_classes_roc_curves(roc):
    """
    Genera y muestra las curvas ROC individuales para cada clase.

    :param pandas.DataFrame roc: DataFrame con los datos de la curva ROC por clase.
    :return: No retorna ningún valor, solo muestra las curvas ROC por clase.
    :rtype: None
    """
    k = roc.shape[0]
    fig, axes = plt.subplots(nrows=(k//3+(k%3>0)), ncols=3)
    axes = axes.flatten()
    labels = ['Diabetes', 'Non-Diabetes'] 
    for i in range(k):
        fpr_dict = {str(k): v for k, v in ast.literal_eval(roc.iloc[i, 3]).items()}
        tpr_dict = {str(k): v for k, v in ast.literal_eval(roc.iloc[i, 4]).items()}
        for j in range(len(labels)):
            roc_auc = auc(fpr_dict[f'{j}'], tpr_dict[f'{j}'])
            axes[i].plot(fpr_dict[f'{j}'], tpr_dict[f'{j}'], lw=2, label=f'{labels[j]} (AUC = {roc_auc:.3f})')      
        axes[i].plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
        axes[i].set_title(f'k = {i+1}')
        axes[i].set_xlabel('1 - Especificidad')
        axes[i].set_ylabel('Sensibilidad')
        axes[i].grid(True)
        axes[i].legend(loc="lower right")
    for i in range(k, len(axes)):
        fig.delaxes(axes[i])    
    plt.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.07)
    plt.subplots_adjust(hspace=0.25, wspace=0.2)
    plt.show()

def plot_metrics(excel_file):
    """
    Genera y muestra visualizaciones clave de las métricas obtenidas durante el entrenamiento de un modelo, incluyendo:
    
    - Matriz de confusión para evaluar el desempeño del modelo en la clasificación.
    - Curvas ROC global y por clase para analizar la capacidad de discriminación del modelo.
    - Evolución del accuracy en entrenamiento y validación a lo largo de las iteraciones.

    :param str excel_file: Ruta del archivo Excel que contiene los resultados del modelo.
    :return: No retorna ningún valor, solo genera y muestra las visualizaciones correspondientes.
    :rtype: None
    """
    data = read_excel_file(excel_file)
    k_metrics, y_test, y_pred, histories, roc = get_data(data)
    plot_confusion_matrix(y_test, y_pred)
    if histories.empty == False:
        plot_accuracy_evolution(histories)
    if roc.empty == False:
        plot_roc_curves(roc)
        #plot_classes_roc_curves(roc)

if __name__ == '__main__':
    excel_file = './' + str(input('Archivo con los datos de las métricas (.xlsx): '))
    plot_metrics(excel_file)