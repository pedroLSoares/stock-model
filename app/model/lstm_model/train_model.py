from app.dto.model_train_parameters import TrainParamsInput
from app.model.lstm_model.LSTM import LSTM
from app.model.data.finance_dataset_generator import get_train_data, load_data
import torch
import mlflow

from app.model.ModelTrainer import ModelTrainer
from app.model.utils import save_trained_model

TICKER = "AMZN"

def run_training(data: TrainParamsInput):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finance_data = load_data(TICKER, "5y")
    X_train, y_train, X_test, y_test, scaler_all, scaler_target = get_train_data(
        finance_data, 
        data.seq_length, 
        data.train_split, 
        device
    )
    model = LSTM(input_size=5, 
                  hidden_size=data.hidden_size, 
                  num_layers=data.num_layers,
                  output_size=1,
                  device=device).to(device)

    trainer = ModelTrainer({
        "learning_rate": data.learning_rate,
        "num_epochs": data.num_epochs
    }, device)
    with mlflow.start_run(run_name="stock_prediction_model"):
        mlflow.log_params(data.model_dump())

        model = trainer.train(model, X_train, y_train)

        trainer.evaluate(model, X_test, y_test, scaler_target)

        save_trained_model(model, scaler_all, scaler_target)

        return model, scaler_all, scaler_target
