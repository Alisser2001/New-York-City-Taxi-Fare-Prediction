FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY train.py .
COPY predict.py .
COPY sample_train.csv .
COPY sample_input.csv .

RUN pip install --no-cache-dir -r requirements.txt

