from sklearn.preprocessing import StandardScaler

def normalizar_datos(x):
    """
    Normaliza los datos utilizando StandardScaler (media 0, desviación típica 1) de Scikit-Learn.

    Args:
        x (Dataframe): Datos a normalizar.

    Returns:
        x_scaled (Dataframe): Datos normalizados.
        scalers (dict): Diccionario de scalers utilizados para cada columna.
    """
    # Selección de variables que se deben escalar (aquellas que no son binarias)
    columnas_escalar = x.columns[x.nunique() > 2]
    # Escalar cada columna individualmente
    scalers = {}
    x_scaled = x.copy()
    for col in columnas_escalar:
        scaler = StandardScaler()
        x_scaled[[col]] = scaler.fit_transform(x_scaled[[col]])
        scalers[col] = scaler  # Guardar el scaler por si se necesita después
    return x_scaled, scalers