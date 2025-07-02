import pickle
import pandas as pd
import numpy as np
from loguru import logger
from typing import Union

def predict(
    trip_distance: float,
    hour: int,
    weekday: int,
    passenger_count: int,
    model_file: str = "model.pkl"
) -> float:
    logger.info(f"Cargando modelo desde {model_file}")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    feature_data = pd.DataFrame({
        'trip_distance': [trip_distance],
        'hour': [hour],
        'weekday': [weekday],
        'passenger_count': [passenger_count]
    })
    
    logger.info(f"Realizando predicción con parámetros: {feature_data.iloc[0].to_dict()}")

    prediction = model.predict(feature_data)
    
    prediction_value = float(prediction[0])
    
    logger.info(f"Predicción realizada: {prediction_value}")
    
    return prediction_value
