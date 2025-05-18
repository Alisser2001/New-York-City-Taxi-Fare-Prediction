import argparse
import os
import pandas as pd
import numpy as np
import pickle
from loguru import logger
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Función para calcular distancia (Haversine)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # km


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True, help='CSV con datos de entrenamiento')
    parser.add_argument('--model_file', required=True, help='Ruta donde guardar el modelo .pkl')
    parser.add_argument('--overwrite_model', action='store_true',
                        help='Sobrescribe el modelo existente si ya existe')
    args = parser.parse_args()

    if os.path.isfile(args.model_file) and not args.overwrite_model:
        logger.error(f"El modelo {args.model_file} ya existe. Usa --overwrite_model para sobrescribir.")
        exit(1)
    elif os.path.isfile(args.model_file):
        logger.info(f"Sobre-escribiendo modelo existente: {args.model_file}")

    logger.info("Cargando datos de entrenamiento")
    df = pd.read_csv(args.data_file, parse_dates=['pickup_datetime'])

    # Limpieza básica
    df = df.dropna(subset=['fare_amount', 'pickup_datetime',
                           'pickup_longitude', 'pickup_latitude',
                           'dropoff_longitude', 'dropoff_latitude',
                           'passenger_count'])
    df = df[(df['fare_amount'] >= 1) & (df['fare_amount'] <= 500)]
    df = df[df['pickup_latitude'].between(40, 42) & df['pickup_longitude'].between(-75, -72)]
    df = df[df['passenger_count'] >= 1]

    logger.info("Extrayendo features de fecha y distancia")
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.weekday
    df['trip_distance'] = haversine(
        df['pickup_longitude'], df['pickup_latitude'],
        df['dropoff_longitude'], df['dropoff_latitude']
    )

    # Definir X e y
    feature_cols = ['trip_distance', 'hour', 'weekday', 'passenger_count']
    logger.info(f"Usando features: {feature_cols}")
    X = df[feature_cols]
    y = df['fare_amount']

    # División en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Configurando y entrenando XGBRegressor")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluación en entrenamiento (opcional)
    score = model.score(X_train, y_train)
    logger.info(f"Score en entrenamiento: {score:.3f}")

    logger.info(f"Guardando modelo en {args.model_file}")
    with open(args.model_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()