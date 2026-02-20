# Stock Prediction API

REST API that predicts the next closing price of a stock using an LSTM neural network. It supports on-demand model retraining in the background and real-time prediction from the last X days of OHLCV (Open, High, Low, Volume, Close) data.

---

## Project overview

- **Model**: LSTM (PyTorch) with configurable hidden size and number of layers.
- **Input**: 60 consecutive days of 5 features per day (Open, High, Low, Volume, Close).
- **Output**: Predicted next-day closing price and trend (Up/Down).
- **Training**: Historical data is fetched via yfinance; you can trigger a new training run via the `/train` endpoint (runs in the background). The serving model is swapped when training finishes.
- **Artifacts**: Trained model weights (`modelo_lstm.pth`) and two MinMaxScalers (`scaler_all.pkl`, `scaler_target.pkl`) are saved under `model_artifacts/` and loaded at startup for prediction.

---

## Architecture & Design Decisions

- **Data Leakage Prevention:** We explicitly use two separate `MinMaxScaler` instances. `scaler_all` fits the input features, while `scaler_target` fits *only* the Close price of the training set. The scaling is applied *after* the train/test split to guarantee the model remains entirely blind to future data distributions.
- **Sliding Window Approach:** The model doesn't just look at isolated days; it learns the sequence of the market using a 60-day rolling window (`seq_length`), inherently capturing temporal dependencies and market momentum.
- **Asynchronous MLOps:** Training is decoupled from inference using FastAPI's `BackgroundTasks`. The API remains responsive to `/predict` calls using the current weights while a new model is trained and dynamically swapped in memory (Hot-Swap) upon completion.

---

## How model training works

1. **Data**
   - Historical data is loaded for a fixed ticker and period (e.g. 5 years) using **yfinance**.
   - Only `Open`, `High`, `Low`, `Volume`, `Close` are used; missing rows are dropped.

2. **Train / test split**
   - Data is split by time (e.g. 80% train, 20% test). No shuffle, to preserve temporal order.

3. **Scaling**
   - **scaler_all**: MinMaxScaler fit on training data for all 5 features (used for model input).
   - **scaler_target**: MinMaxScaler fit on training **Close** only (used to scale/inverse-scale the target and predictions).

4. **Sequences**
   - For each position `i`, the input is a window of `seq_length` days (default 60) and the target is the **Close** of the next day.
   - Sequences are built from the scaled data; train and test sets become PyTorch tensors.

5. **LSTM training**
   - **Model**: LSTM with `input_size=5`, configurable `hidden_size` and `num_layers`, and a linear layer that outputs one value (next Close).
   - **Loss**: MSE. Optimizer: Adam with configurable learning rate.
   - Training runs for a fixed number of epochs; metrics (e.g. train loss, test RMSE/MAPE) can be logged to MLflow when available.

6. **Saving**
   - After training, the script saves:
     - Model state dict → `model_artifacts/modelo_lstm.pth`
     - **scaler_all** → `model_artifacts/scaler_all.pkl`
     - **scaler_target** → `model_artifacts/scaler_target.pkl`
   - When training is triggered via the API, it runs in a background task; when it finishes, the in-memory model and scalers are replaced so the next predictions use the new model.

---

## How prediction works

1. **Request**
   - Client sends a JSON body with a list of **60 days**, each day with exactly 5 values: `[Open, High, Low, Volume, Close]` (same order as in training).

2. **Preprocessing**
   - The 60×5 array is transformed with the loaded **scaler_all** (same MinMax scaling as in training).
   - The result is converted to a PyTorch tensor and passed to the LSTM (batch size 1).

3. **Model**
   - The LSTM outputs a single scalar (next Close in scaled space). The API then applies **scaler_target.inverse_transform** to get the predicted price in original units.

4. **Response**
   - The API returns:
     - `last_known_close`: Close of the last day in the request (index 4 of the last row).
     - `predicted_next_close`: Predicted next closing price.
     - `trend`: `"Up"` if predicted close > last known close, otherwise `"Down"`.
     - `model_version`: Incremented each time a background training run completes (useful to know which model served the response).

If the model or scalers are not loaded (e.g. missing artifacts), prediction requests will fail until a trained model is available.

---

## API endpoints

| Method | Path             | Description |
|--------|------------------|-------------|
| GET    | `/`              | Health message. |
| GET    | `/system-health` | CPU and RAM usage; indicates if a training run is active. |
| GET    | `/model-metrics` | Returns the latest training run from MLflow (run_id, parameters, metrics). |
| POST   | `/train`         | Starts LSTM training in the background (body: training hyperparameters). Only one training run at a time. |
| POST   | `/predict`       | Sends 60 days of OHLCV; returns predicted next close, trend, and model version. |
| GET    | `/metrics`       | Prometheus metrics (if instrumentator is enabled). |

Interactive docs: `http://localhost:8000/docs`.

---

## Run locally (without Docker)

From the **project root**:

```bash
# Create venv, install dependencies, then:
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Ensure `app/model_artifacts/` (or the path used in code) contains the trained model and scalers, or run a training job first (e.g. via `/train` or the training script).

---

## Container (Docker)

### Build

From the **project root** (where `app/`, `container/` are):

```bash
docker build -t stock-model .
```

### Run

```bash
docker run -p 8000:8000 -v ${PWD}/app/model_artifacts:/app/app/model_artifacts stock-model
```

API: `http://localhost:8000`. Docs: `http://localhost:8000/docs`.
