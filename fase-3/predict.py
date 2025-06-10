import argparse
import os
import pandas as pd
import numpy as np
import pickle
from loguru import logger

# Función Haversine para calcular distancias entre coordenadas
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='CSV con datos raw o ya procesados')
    parser.add_argument('--predictions_file', required=True, help='CSV donde guardar las predicciones')
    parser.add_argument('--model_file', required=True, help='Archivo .pkl con el modelo entrenado')
    args = parser.parse_args()

    # Validación de existencia de archivos
    if not os.path.isfile(args.model_file):
        logger.error(f"No existe el archivo de modelo: {args.model_file}")
        exit(1)
    if not os.path.isfile(args.input_file):
        logger.error(f"No existe el archivo de entrada: {args.input_file}")
        exit(1)

    # Cargar datos de entrada sin parsear fechas automático
    logger.info("Cargando datos de entrada")
    df = pd.read_csv(args.input_file)

    # Determinar tipo de input: raw (con pickup_datetime) o procesado (solo features)
    feature_cols = ['trip_distance', 'hour', 'weekday', 'passenger_count']
    if 'pickup_datetime' in df.columns:
        # Preprocesamiento completo
        logger.info("Modo raw detectado: parsing fechas y calculando features")
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['trip_distance'] = haversine(
            df['pickup_longitude'], df['pickup_latitude'],
            df['dropoff_longitude'], df['dropoff_latitude']
        )
        df['hour'] = df['pickup_datetime'].dt.hour
        df['weekday'] = df['pickup_datetime'].dt.weekday
        X = df[feature_cols]
    else:
        # Asumimos que el CSV ya contiene las columnas de features
        logger.info("Modo procesado detectado: usando columnas de features directamente")
        missing = set(feature_cols) - set(df.columns)
        if missing:
            logger.error(f"Faltan columnas de features: {missing}")
            exit(1)
        X = df[feature_cols]

    # Cargar modelo entrenado
    logger.info(f"Cargando modelo desde {args.model_file}")
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Generar predicciones
    logger.info("Generando predicciones")
    preds = model.predict(X)

    # Preparar output: si hay columna 'key', la incluimos
    output = pd.DataFrame({'prediction': preds})
    if 'key' in df.columns:
        output.insert(0, 'key', df['key'])

    # Guardar predicciones
    logger.info(f"Guardando predicciones en {args.predictions_file}")
    output.to_csv(args.predictions_file, index=False)

if __name__ == '__main__':
    main()