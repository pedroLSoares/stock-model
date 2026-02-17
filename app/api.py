import time
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient
import torch
import uvicorn
import numpy as np
import joblib

from app.dto.model_train_parameters import TrainParamsInput
from app.model.lstm_model import train_model
from app.model.lstm_model.LSTM import LSTM
from app.dto.stock_input import StockInput
from app import logger
import psutil
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="API to predict stock's next price",
    description=""
)
Instrumentator().instrument(app).expose(app)
model = None
scaler_all = None
scaler_target = None
model_version = 0

is_training = False

@app.on_event("startup")
def load_artifacts():
    global model, scaler_all, scaler_target
    
    try:
        scaler_all = joblib.load("app/model_artifacts/scaler_all.pkl")
        scaler_target = joblib.load("app/model_artifacts/scaler_target.pkl")

        model = LSTM(input_size=5, hidden_size=50, num_layers=2, output_size=1)
        
        model.load_state_dict(torch.load("app/model_artifacts/modelo_lstm.pth", map_location=torch.device('cpu')))
        model.eval() 
        
        logger.info("Model and scalers loaded successfully.")
        
    except FileNotFoundError as e:
        logger.error(f"Critical error: File not found. {e}")


@app.get("/system-health", tags=["Monitoring"])
def get_system_health():
    """
    Returns real-time server resource usage (CPU and RAM).
    Useful for monitoring scalability during model retraining.
    """
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    
    return {
        "status": "online",
        "cpu_usage_percent": cpu_usage,
        "ram_usage_percent": ram.percent,
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "is_training_active": is_training 
    }


@app.get("/")
def home():
    return {"message": "API is running!"}



def execute_training(data: TrainParamsInput):
    global is_training, model, scaler_all, scaler_target, model_version
    try:
        newModel, new_scaler_all, new_scaler_target = train_model.run_training(data=data, savemodel=False)

        model = newModel
        scaler_all = new_scaler_all
        scaler_target = new_scaler_target
        model_version = model_version + 1
    except Exception as e:
        logger.error(f"[BACKGROUND] Training failed: {e}")
    finally:
        is_training = False


@app.post("/train")
def train_endpoint(payload: TrainParamsInput, background_tasks: BackgroundTasks):
    """
    Starts model training for a given ticker in the background.
    """
    global is_training
    
    if is_training:
        raise HTTPException(
            status_code=429, 
            detail="A training run is already in progress. Please wait for it to finish."
        )
    
    is_training = True
    background_tasks.add_task(execute_training, data=payload)
    
    return {
        "message": f"Training for model started in background.",
        "status": "processing",
        "tip": "You can keep using /predict with the current model until the new one is ready."
    }

@app.get("/model-metrics", summary="Returns the latest model training metrics from MLflow")
def get_model_metrics():
    """
    Return parameters and metrics of the most recent training run from MLflow.

    Looks up the experiment by name (Tech_Challenge_LSTM), fetches the latest
    run by start_time, and returns its run_id, params, and metrics. If the
    experiment or run does not exist, returns a status message instead.
    On error (e.g. MLflow unavailable), returns an error message.
    """
    try:
        client = MlflowClient()
        
        experiment = client.get_experiment_by_name("Tech_Challenge_LSTM")
        
        if not experiment:
            return {"status": "No metrics yet. Run training first."}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            return {"status": "No training runs found in MLflow."}

        last_run = runs[0]

        return {
            "train_id": last_run.info.run_id,
            "parameters": last_run.data.params,
            "results": last_run.data.metrics
        }
    except Exception as e:
        return {"error": f"Failed to read MLflow: {str(e)}"}

@app.post("/predict")
def predict_stock(data: StockInput):
    """
    Accepts a JSON with the ticker and a list of 60 historical prices.
    Returns the predicted next price.
    """

    if model is None or scaler_all is None or scaler_target is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please wait for the background training to finish or execute model training."
        )
    
    if len(data.features[0]) != 5:
         raise HTTPException(
            status_code=400, 
            detail=f"Each day must have 5 values [Open, High, Low, Volume, Close]. Received: {len(data.features[0])}"
        )

    try:
        input_array = np.array(data.features)
    
        scaled_input = scaler_all.transform(input_array)
        tensor_input = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = model(tensor_input).cpu().numpy()
        
        prediction_real = scaler_target.inverse_transform(prediction_scaled)[0][0]
        
        last_close_price = data.features[-1][4]

        return {
            "last_known_close": last_close_price,
            "predicted_next_close": round(float(prediction_real), 2),
            "trend": "Up" if prediction_real > last_close_price else "Down",
            "model_version": model_version
        }

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)