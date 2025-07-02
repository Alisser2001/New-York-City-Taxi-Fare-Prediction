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
def predict():
    result = subprocess.run(
        ["python", "default-predict.py", "--input_file", "sample_input.csv", "--predictions_file", "preds.csv", "--model_file", "model.pkl"],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr":result.stderr}

@app.get("/predict")
def predict(
    trip_distance: float,
    hour: int,
    weekday: int,
    passenger_count: int,
    model_file: str = "model.pkl"
):
    try:
        if not os.path.isfile(model_file):
            raise HTTPException(status_code=404, detail=f"Archivo de modelo no encontrado: {model_file}")
        
        if hour < 0 or hour > 23:
            raise HTTPException(status_code=400, detail="hour debe estar entre 0 y 23")
        if weekday < 0 or weekday > 6:
            raise HTTPException(status_code=400, detail="weekday debe estar entre 0 y 6")
        if passenger_count < 1:
            raise HTTPException(status_code=400, detail="passenger_count debe ser mayor a 0")
        if trip_distance < 0:
            raise HTTPException(status_code=400, detail="trip_distance debe ser mayor o igual a 0")
        
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
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")