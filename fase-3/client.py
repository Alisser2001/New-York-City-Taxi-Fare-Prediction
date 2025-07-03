import requests

# Endpoint 1: Train
# Endpoint para correr el proceso de entrenamiento con los datos de prueba en sample_train.csv
# Y generar el archivo model.pkl
endpoint = 'http://localhost:8001/train'
response = requests.post(endpoint)
print("train:", response.json())

# Endpoint 2: Default predict
# Endpoint para correr un proceso de prediccion con los datos de prueba en sample_input.csv
# Y generar el archivo predictions.csv con los resultados de las predicciones
endpoint = 'http://localhost:8001/default-predict'
response = requests.post(endpoint)
print("default-predict:", response.json())

# Endpoint 3: Predict con parámetros
# Endpoint para ingresar datos de prueba para un proceso de prediccion
# Y recibir una prediccion sobre el costo total del viaje segun los parametros ingresadoss
endpoint = 'http://localhost:8001/predict?trip_distance=1.03&hour=17&weekday=0&passenger_count=1'
response = requests.get(endpoint)
print("predict:", response.json())
