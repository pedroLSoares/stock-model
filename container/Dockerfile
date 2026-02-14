# Stock price prediction API
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    numpy==2.4.1 \
    joblib==1.5.3 \
    pydantic==2.12.5 \
    scikit-learn==1.8.0 \
    python-dotenv==1.2.1

COPY app ./app
RUN mkdir -p model_artifacts

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
