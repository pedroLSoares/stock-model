from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import torch


def load_data(ticker: str, period: str):
    hist = yf.Ticker(ticker).history(period=period)
    hist = hist[['Open', 'High', 'Low', 'Volume', 'Close']]

    hist = hist.dropna()
    return hist.values


def create_sequences(data, seq_length, target_idx):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_idx] 
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys).reshape(-1, 1)


def get_train_data(data, seq_length, train_split, device):
    train_size = int(len(data) * train_split)
    train_data_raw = data[:train_size]
    test_data_raw = data[train_size:]


    scaler_all = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_all.fit_transform(train_data_raw)
    test_scaled = scaler_all.transform(test_data_raw)


    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(train_data_raw[:, -1].reshape(-1, 1))

    X_train, y_train = create_sequences(train_scaled, seq_length, target_idx=-1)
    X_test, y_test = create_sequences(test_scaled, seq_length, target_idx=-1)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_all, scaler_target