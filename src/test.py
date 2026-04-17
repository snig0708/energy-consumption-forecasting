import numpy as np
import pandas as pd
import holidays

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# CONFIG
# ======================
DATA_PATH = "../data/PJME_hourly.csv"
MODEL_PATH = "../saved_models/best_model_v2.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD CHECKPOINT
# ======================
checkpoint = torch.load(MODEL_PATH, map_location=device)
BEST_MODEL = checkpoint["model_name"]
SEQ_LENGTH = checkpoint["seq_length"]
feature_cols = checkpoint["feature_cols"]

print("Best model from checkpoint:", BEST_MODEL)

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "Datetime": "datetime",
    "PJME_MW": "value"
})

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
df = df.set_index("datetime")

# ======================
# FEATURE ENGINEERING
# ======================
df_features = (
    df
    .assign(hour=df.index.hour)
    .assign(day=df.index.day)
    .assign(month=df.index.month)
    .assign(day_of_week=df.index.dayofweek)
    .assign(week_of_year=df.index.isocalendar().week.astype(int))
)

def generate_cyclical_features(df, col_name, period, start_num=0):
    df = df.copy()
    df[f"sin_{col_name}"] = np.sin(2 * np.pi * (df[col_name] - start_num) / period)
    df[f"cos_{col_name}"] = np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    return df.drop(columns=[col_name])

df_features = generate_cyclical_features(df_features, "hour", 24, 0)
df_features = generate_cyclical_features(df_features, "day_of_week", 7, 0)
df_features = generate_cyclical_features(df_features, "month", 12, 1)
df_features = generate_cyclical_features(df_features, "week_of_year", 52, 0)

us_holidays = holidays.US()

def is_holiday(ts):
    return 1 if ts.normalize() in us_holidays else 0

df_features["is_holiday"] = df_features.index.to_series().apply(is_holiday)

# ======================
# SPLIT + SCALE
# ======================
train_size = int(len(df_features) * 0.8)
train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

scaler = MinMaxScaler()
train_values = scaler.fit_transform(train_df[feature_cols])
test_values = scaler.transform(test_df[feature_cols])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y).reshape(-1, 1)

X_test, y_test = create_sequences(test_values, SEQ_LENGTH)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

INPUT_SIZE = X_test.shape[2]
HIDDEN_SIZE = 32
NUM_LAYERS = 2

# ======================
# MODELS
# ======================
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])

if BEST_MODEL == "RNN":
    model = RNNModel()
elif BEST_MODEL == "LSTM":
    model = LSTMModel()
else:
    model = GRUModel()

model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

def inverse_transform_target(values):
    temp = np.zeros((len(values), len(feature_cols)))
    temp[:, 0] = values.reshape(-1)
    return scaler.inverse_transform(temp)[:, 0].reshape(-1, 1)

preds, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        preds.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

preds = inverse_transform_target(np.array(preds))
actuals = inverse_transform_target(np.array(actuals))

mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")