from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.post("/train")
def train():
    result = subprocess.run(
        ["python", "train.py", "--data_file", "sample_train.csv", "--model_file", "model.pkl", "--overwrite_model"],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr": result.stderr}

@app.post("/predict")
def predict():
    result = subprocess.run(
        ["python", "predict.py", "--input_file", "sample_input.csv", "--predictions_file", "preds.csv", "--model_file", "model.pkl"],
        capture_output=True,
        text=True
    )
    return {"stdout": result.stdout, "stderr":result.stderr}