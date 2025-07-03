from fastapi import FastAPI, HTTPException
from predict import predict
import subprocess
import os

# Crear la aplicación FastAPI
app = FastAPI()

# Endpoint POST para entrenar el modelo
@app.post("/train")
def train():
    # Ejecuta el script de entrenamiento usando subprocess
    result = subprocess.run(
        ["python", "train.py", "--data_file", "sample_train.csv", "--model_file", "model.pkl", "--overwrite_model"],
        capture_output=True,
        text=True
    )
    # Retorna tanto la salida estándar como los errores
    return {"stdout": result.stdout, "stderr": result.stderr}

# Endpoint POST para hacer predicciones usando un archivo CSV base
@app.post("/default-predict")
def default_predict():
    # Ejecuta el script de predicción por lotes usando subprocess
    result = subprocess.run(
        ["python", "default-predict.py", "--input_file", "sample_input.csv", "--predictions_file", "preds.csv", "--model_file", "model.pkl"],
        capture_output=True,
        text=True
    )
    # Retorna la salida del comando ejecutado
    return {"stdout": result.stdout, "stderr":result.stderr}

# Endpoint GET para hacer predicciones individuales en tiempo real
@app.get("/predict")
def fetch_predict(
    trip_distance: float, # Distancia del viaje
    hour: int, # Hora del día (0-23)
    weekday: int, # Día de la semana (0-6)
    passenger_count: int, # Número de pasajeros
    model_file: str = "model.pkl"
): 
    # Validar que el archivo del modelo existe
    if not os.path.isfile(model_file):
        return {"error": f"Archivo de modelo no encontrado: {model_file}", "prediction": None}
    
    # Validaciones de los parámetros de entrada
    if hour < 0 or hour > 23:
        return {"error": "hour debe estar entre 0 y 23", "prediction": None}
    if weekday < 0 or weekday > 6:
        return {"error": "weekday debe estar entre 0 y 6", "prediction": None}
    if passenger_count < 1:
        return {"error": "passenger_count debe ser mayor a 0", "prediction": None}
    if trip_distance < 0:
        return {"error": "trip_distance debe ser mayor o igual a 0", "prediction": None}
    
    try:
        # Llamar a la función de predicción importada
        prediction = predict(
            trip_distance=trip_distance,
            hour=hour,
            weekday=weekday,
            passenger_count=passenger_count,
            model_file=model_file
        )
        
        # Retornar la predicción exitosa con los parámetros usados
        return {
            "prediction": prediction,
            "input_parameters": {
                "trip_distance": trip_distance,
                "hour": hour,
                "weekday": weekday,
                "passenger_count": passenger_count
            },
            "error": None
        }
        
    except Exception as e:
        # Manejar cualquier error durante la predicción
        return {
            "error": f"Error en la predicción: {str(e)}",
            "prediction": None,
            "input_parameters": {
                "trip_distance": trip_distance,
                "hour": hour,
                "weekday": weekday,
                "passenger_count": passenger_count
            }
        }