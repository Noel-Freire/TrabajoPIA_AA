import pandas as pd

def cargar_datos(path,target):
    """
    Carga los datos de un archivo CSV.
    
    Args:
        path (str): Ruta de los datos a normalizar.
        target (str): Nombre de la columna que contiene las etiquetas.

    Returns:
        X (Dataframe): Datos que se enviarán al modelo.
        y (Series): Etiquetas de cada dato.
    """
    # Cargar el dataset
    datos = pd.read_csv(path)
    x = datos.drop(columns=target)
    y = datos[target]
    print(f'Número de variables: {x.shape[1]}')
    print(f'Número total de datos: {x.shape[0]}')
    # Mostrar información del dataset
    n_clases = y.value_counts()
    for clase, n in n_clases.items():
        print(f'Datos de la clase {clase}: {n}') 
    return x, y