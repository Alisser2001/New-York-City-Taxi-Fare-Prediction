-----
## **🚖 Predicción de Tarifas de Taxi en Nueva York con XGBoost**

**📘 Descripción**

Este proyecto se desarrollado para la materia **Modelos y Simulación de Sistemas I** - Ingenieria de Sistemas - Universidad de Antioquia. El objetivo es predecir la tarifa de viajes en taxi en la ciudad de Nueva York utilizando técnicas de aprendizaje automático, específicamente el algoritmo **XGBoost**.

La competencia seleccionada es:

- [New York City Taxi Fare Prediction - Kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)

**👥 Autores**

- Alexander Valencia Delgado - alexander.valenciad@udea.edu.co

- Juan Esteven Carmona Muñoz - estiven.carmona@udea.edu.co

**Docente:** 

Raúl Ramos Pollán


**📁 Estructura del Proyecto**

├── **modelo\_xgboost.ipynb**

├── **train2.csv**

├── **test.csv**

├──**README.md**

- **modelo\_xgboost.ipynb:** Notebook principal con todo el flujo del proyecto.
- **train2.csv:** Subconjunto del conjunto de entrenamiento original.
- **test.csv** Conjunto de datos de prueba proporcionado por Kaggle.
- **README.md:** Este archivo con instrucciones detalladas.

**🛠️ Requisitos**

Debemos tener instaladas las siguientes bibliotecas:

`pip install pandas numpy scikit-learn xgboost matplotlib ipywidgets joblib`

**🚀 Instrucciones de Ejecución**

**1. Montamos Google Drive (para usar los datos en Google Colab)**

```python
from google.colab import drive

drive.mount('/content/drive')
```

Debes colocar la ruta donde este el archivo train2.csv  en el notebook:

```python
/content/drive/MyDrive/.../train2.csv
```

**2. Cargar y Preprocesar los Datos**

- Cargamos el archivo train2.csv.
- Eliminamos valores nulos.
- Filtramos outliers en fare\_amount, coordenadas y passenger\_count.
- Calculamos la distancia del viaje utilizando la fórmula de **Haversine**.
- Extraemos características temporales: hora y día de la semana.

**3. Entrenar el Modelo**

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

**4. Generar Predicciones**

- Cargamos test.csv.
- Aplicamos el mismo preprocesamiento que al conjunto de entrenamiento.
- Realizamos predicciones con el modelo entrenado.
- Guardamos las predicciones en submission.csv con las columnas key y fare_amount.

**5. Visualizar Resultados**

- Graficamos las predicciones frente a los valores reales del conjunto de validación.
- Mostramos la importancia de las características utilizando xgb.plot_importance(model).

**6. Interfaz Interactiva**

- Utilizamos ipywidgets para crear una interfaz que permita ingresar:
  - Distancia del viaje (trip\_distance)
  - Fecha y hora de recogida (pickup\_datetime)
  - Número de pasajeros (passenger\_count)
- Al hacer clic en el botón "Predecir", se mostrará la tarifa estimada.

**7. Guardar y Cargar el Modelo**

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

**📊 Resultados**

- **RMSE en validación**: Aproximadamente entre 3.5 y 4.5 USD.
- **Archivo de predicciones**: submission.csv listo.
- **Interfaz interactiva**: Permite obtener predicciones personalizadas de tarifas.

**🙌 Créditos**

Este proyecto se basa en la solución desarrollada por [rrkcoder en Kaggle](https://www.kaggle.com/code/rrkcoder/xgboost/notebook). A partir de su trabajo, se realizaron adaptaciones y mejoras, incluyendo la implementación de widgets interactivos para facilitar la entrada de datos y la generación de predicciones personalizadas.

**📌 Notas Adicionales**

- Asegúrarnos de seguir los pasos en el orden indicado para evitar errores.
- Debemos verificar que los archivos train2.csv y test.csv estén en las rutas correctas.
- Si encuentras algún problema o tienes preguntas, no dudes en consultarnos.
-----


-----
