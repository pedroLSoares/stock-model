import os
import torch
import joblib
import mlflow
from app import logger


def save_trained_model(model, scaler_all, scaler_target, path="app/model_artifacts"):
    os.makedirs(path, exist_ok=True)
    

    model_path = os.path.join(path, "modelo_lstm.pth")
    scaler_all_path = os.path.join(path, "scaler_all.pkl")
    scaler_target_path = os.path.join(path, "scaler_target.pkl")

    torch.save(model.state_dict(), model_path)

    joblib.dump(scaler_all, scaler_all_path)
    joblib.dump(scaler_target, scaler_target_path)
    
    logger.info("Local artifact files saved successfully.")

    if mlflow.active_run():
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_all_path)
        mlflow.log_artifact(scaler_target_path)