FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    joblib \
    pydantic \
    scikit-learn \
    pandas \
    yfinance \
    mlflow \
    psutil \
    prometheus-fastapi-instrumentator \
    python-dotenv

COPY app ./app

RUN mkdir -p app/model_artifacts app/mlruns

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]