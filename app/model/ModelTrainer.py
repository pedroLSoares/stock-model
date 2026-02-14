import torch
import torch.nn as nn
import numpy as np
import mlflow
from app import logger


class ModelTrainer:
    def __init__(self, params: dict, device: torch.device):
        self.params = params
        self.device = device
        self.criterion = nn.MSELoss()
        
    def train(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor):
        logger.info(f"Starting training with {len(X_train)} samples...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params["learning_rate"])
        
        for epoch in range(self.params["num_epochs"]):
            model.train()
            
            outputs = model(X_train)
            loss = self.criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if mlflow.active_run():
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{self.params["num_epochs"]}], Loss (MSE): {loss.item():.6f}')
                
        logger.info("Training completed.")
        return model

    def evaluate(self, model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, scaler_target):
        logger.info("Evaluating model on test data...")
        model.eval()
        
        with torch.no_grad():
            test_predictions = model(X_test).cpu().numpy()
            y_test_real = y_test.cpu().numpy()
        
        real_pred = scaler_target.inverse_transform(test_predictions)
        real_actual = scaler_target.inverse_transform(y_test_real)
        
        rmse = np.sqrt(np.mean((real_actual - real_pred) ** 2))
        mape = np.mean(np.abs((real_actual - real_pred) / real_actual)) * 100
        
        logger.info(f"Final RMSE: {rmse:.2f}")
        logger.info(f"Final MAPE: {mape:.2f}%")
        
        if mlflow.active_run():
            mlflow.log_metric("test_rmse_real", rmse)
            mlflow.log_metric("test_mape_real", mape)
            
        return rmse, mape