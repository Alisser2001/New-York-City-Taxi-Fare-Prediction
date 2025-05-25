# Imagen base liviana de Python 3.10 para reducir el tamaño del contenedor
FROM python:3.10-slim 
# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias a la raíz del directorio de trabajo
COPY requirements.txt .
# Copia el script de entrenamiento al contenedor
COPY train.py .
# Copia el script de predicción al contenedor
COPY predict.py .
# Copia un archivo de muestra para entrenamiento
COPY sample_train.csv .
# Copia un archivo de muestra para predicción
COPY sample_input.csv .

# Instala las dependencias listadas, sin usar caché para reducir el tamaño final
RUN pip install --no-cache-dir -r requirements.txt

