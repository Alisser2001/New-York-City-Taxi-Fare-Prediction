from fastapi import FastAPI, HTTPException
from predict import predict
import subprocess
import os

app = FastAPI()

@app.post("/train")
def train():
    result = subprocess.run(
        ["python", "train.py", "--data_file", "sample_train.csv", "--model_file", "model.pkl", "--overwrite_model"],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr": result.stderr}

@app.post("/default-predict")
def default_predict():
    result = subprocess.run(
        ["python", "default-predict.py", "--input_file", "sample_input.csv", "--predictions_file", "preds.csv", "--model_file", "model.pkl"],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr":result.stderr}

@app.get("/predict")
def fetch_predict(
    trip_distance: float,
    hour: int,
    weekday: int,
    passenger_count: int,
    model_file: str = "model.pkl"
):
    if not os.path.isfile(model_file):
        return {"error": f"Archivo de modelo no encontrado: {model_file}", "prediction": None}
    
    if hour < 0 or hour > 23:
        return {"error": "hour debe estar entre 0 y 23", "prediction": None}
    if weekday < 0 or weekday > 6:
        return {"error": "weekday debe estar entre 0 y 6", "prediction": None}
    if passenger_count < 1:
        return {"error": "passenger_count debe ser mayor a 0", "prediction": None}
    if trip_distance < 0:
        return {"error": "trip_distance debe ser mayor o igual a 0", "prediction": None}
    
    try:
        prediction = predict(
            trip_distance=trip_distance,
            hour=hour,
            weekday=weekday,
            passenger_count=passenger_count,
            model_file=model_file
        )
        
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