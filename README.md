-----
## **🚖 Predicción de Tarifas de Taxi en Nueva York con XGBoost**

**📘 Descripción**

Este proyecto se desarrollado para la materia **Modelos y Simulación de Sistemas I** - Ingenieria de Sistemas - Universidad de Antioquia. El objetivo es predecir la tarifa de viajes en taxi en la ciudad de Nueva York utilizando técnicas de aprendizaje automático, específicamente el algoritmo **XGBoost**.

La competencia seleccionada es:

- [New York City Taxi Fare Prediction - Kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)

**👥 Autores**

- Alexander Valencia Delgado - alexander.valenciad@udea.edu.co

- Juan Estiven Carmona Muñoz - estiven.carmona@udea.edu.co

**Docente:** 

Raúl Ramos Pollán


**📁 Estructura del Proyecto**

├── **modelo\_xgboost.ipynb**

├── **train.csv y test.csv** Se deben descargar de Kaggle en [Datos Kaggle](https://drive.google.com/drive/folders/1v9n0fnIAC4OZ1sdhGYZ29yM8gs8aMovB?usp=sharing "Datos")

├──**README.md**

- **modelo\_xgboost.ipynb:** Notebook principal con todo el flujo del proyecto.
- **train.csv:** Subconjunto del conjunto de entrenamiento original.
- **test.csv** Conjunto de datos de prueba proporcionado por Kaggle.
- **README.md:** Este archivo con instrucciones detalladas.

**🛠️ Requisitos**

Debemos tener instaladas las siguientes bibliotecas:

`pip install pandas numpy scikit-learn xgboost matplotlib ipywidgets joblib`


-----
## **🚖 FASE 1**


**🚀 Instrucciones de Ejecución**

**1. Descargamos train.cvs y test.cvs de la plataforma kaggle** [Datos Kaggle](https://drive.google.com/drive/folders/1v9n0fnIAC4OZ1sdhGYZ29yM8gs8aMovB?usp=sharing "Datos Kaggle")

**2. Descargamos Modelo_XGBoost.ipynb** [Modelo_XGBoost.ipynb](https://github.com/alexvadelgado/New-York-City-Taxi-Fare-Prediction/blob/main/Fase%201/Modelo_XGBoost.ipynb "Modelo_XGBoost.ipynb")

**3. Cargamos Modelo_XGBoost.ipynb en el drive de google y lo ejecutamos**

**4. Cargamos el archivo train.csv**

```python
from google.colab import files
uploaded = files.upload()
train_df = pd.read_csv('train.csv')
```

Debes colocar la ruta donde este el archivo train.csv  en el pc:


**5. Cargar y Preprocesar los Datos**

- Cargamos el archivo train.csv.
- Eliminamos valores nulos.
- Filtramos outliers en fare\_amount, coordenadas y passenger\_count.
- Calculamos la distancia del viaje utilizando la fórmula de **Haversine**.
- Extraemos características temporales: hora y día de la semana.

**6. Entrenar el Modelo**

- Dividimos los datos en conjuntos de entrenamiento y validación (80/20).
- Entrenamos un modelo XGBRegressor con los siguientes hiperparámetros:

```python
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,       # Número de árboles
    max_depth=5,             # Profundidad máxima de cada árbol
    learning_rate=0.1,       # Tasa de aprendizaje
    subsample=0.8,           #% de datos usados en cada árbol
    random_state=42,       # Semilla para reproducibilidad
    n_jobs=-1                    # Usar todos los núcleos del CPU
)
```

- Evaluamos el modelo utilizando RMSE.

**7. Generar Predicciones**

- Cargamos test.csv.
- Aplicamos el mismo preprocesamiento que al conjunto de entrenamiento.
- Realizamos predicciones con el modelo entrenado.
- Guardamos las predicciones en submission.csv con las columnas key y fare_amount.

**8. Visualizar Resultados**

- Graficamos las predicciones frente a los valores reales del conjunto de validación.
- Mostramos la importancia de las características utilizando xgb.plot_importance(model).

**9. Interfaz Interactiva**

- Utilizamos ipywidgets para crear una interfaz que permita ingresar:
  - Distancia del viaje (trip\_distance)
  - Fecha y hora de recogida (pickup\_datetime)
  - Número de pasajeros (passenger\_count)
- Al hacer clic en el botón "Predecir", se mostrará la tarifa estimada.

**10. Guardar y Cargar el Modelo**

- Guardamos el modelo entrenado:

```python
import joblib
modelo_entrenado = model.fit(X_train, y_train)
joblib.dump(modelo_entrenado, 'modelo_taxi.pkl')
```

- Cargamos el modelo guardado:

```python
modelo_cargado = joblib.load('modelo_taxi.pkl')
```

**📊 Resultados FASE 1 **

- **RMSE en validación**: Aproximadamente entre 3.5 y 4.5 USD.
- **Archivo de predicciones**: submission.csv listo.
- **Interfaz interactiva**: Permite obtener predicciones personalizadas de tarifas.


-----
## **🚖 FASE 2**


**🚀 Descripción**

Esta fase del proyecto consiste en el despliegue de un modelo de predicción de tarifas de taxi en la ciudad de Nueva York mediante el uso de contenedores Docker. El contenedor incluye todos los componentes necesarios para:

- Entrenar un nuevo modelo con datos personalizados (`train.py`)
- Generar predicciones a partir de un archivo CSV (`predict.py`)

**📁 Estructura del Directorio**

├── **.dockerignore**

├── **Dockerfile** 

├── **model.pkl** (opcional, generado al entrenar).

├── **predict.py** 

├── **predictions.csv** (generado al predecir).

├── **requirements.txt**

├── **sample_input.csv**

├── **sample_train.csv** --> Se debe descargar del Drive en [proyecto taxi kaggle](https://drive.google.com/file/d/1yJk6KRHS0agNJWfboiy3ieQrYVuNoPMC/view?usp=sharing).

├── **train.py** 


## ⚙️ Requisitos previos

- Docker instalado
- Python 3.8+ 

## 🐳 Construcción de la imagen Docker

Para construir la imagen Docker (desde el directorio fase-2):

    docker build -t nyc-taxi-model .

## 🧠 Entrenamiento del modelo

El script `train.py` permite entrenar un modelo de predicción desde un conjunto de datos CSV.

### 🔧 Comando

`docker run --rm -v $(pwd):/app nyc-taxi-model python train.py --data_file sample_train.csv --model_file model.pkl --overwrite_model` 

-   `--data_file`: archivo CSV con los datos de entrenamiento (debe contener las columnas necesarias).
    
-   `--model_file`: ruta donde se guardará el modelo entrenado.
    
-   `--overwrite_model`: sobrescribe el modelo si ya existe.
    

## 🔍 Generación de predicciones

El script `predict.py` permite generar predicciones desde un archivo CSV de entrada.

### 🔧 Comando

`docker run --rm -v $(pwd):/app nyc-taxi-model python predict.py --input_file sample_input.csv --model_file model.pkl --predictions_file predictions.csv` 

-   `--input_file`: archivo CSV con datos de entrada.
    
-   `--predictions_file`: archivo CSV donde se guardarán las predicciones.
    
-   `--model_file`: archivo `.pkl` del modelo previamente entrenado.
    

## 🛠️ Comentarios técnicos

### `train.py`

-   Carga datos desde un CSV, limpia valores atípicos y faltantes.
    
-   Extrae características relevantes: hora, día de la semana y distancia (usando la fórmula Haversine).
    
-   Entrena un modelo `XGBRegressor` de XGBoost.
    
-   Guarda el modelo serializado como `.pkl`.
    

### `predict.py`

-   Permite trabajar con datos crudos (con `pickup_datetime`) o ya procesados (features listas).
    
-   Calcula los mismos features que en entrenamiento.
    
-   Usa el modelo `.pkl` para hacer predicciones y guarda los resultados en CSV.
    

**🙌 Créditos**

Este proyecto se basa en la solución desarrollada por [rrkcoder en Kaggle](https://www.kaggle.com/code/rrkcoder/xgboost/notebook). A partir de su trabajo, realizamos adaptaciones y mejoras, incluimos la implementación de widgets interactivos para facilitar la entrada de datos y la generación de predicciones personalizadas.

**📌 Notas Adicionales**

- Asegúrarnos de seguir los pasos en el orden indicado para evitar errores.
- Debemos verificar que los archivos train.csv, sample_train.csv y test.csv estén en las rutas correctas.
- Si encuentras algún problema o tienes preguntas, no dudes en consultarnos.
-----


-----
