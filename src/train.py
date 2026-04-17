import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================
# CONFIG
# ======================
SEQ_LENGTH = 24
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

DATA_PATH = "../data/PJME_hourly.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

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
# VISUALIZE
# ======================
plt.figure(figsize=(15, 5))
plt.plot(df_features.index, df_features["value"])
plt.title("Energy Consumption")
plt.show()

# ======================
# SPLIT
# ======================
train_size = int(len(df_features) * 0.8)
train_df = df_features.iloc[:train_size]
test_df = df_features.iloc[train_size:]

# ======================
# SCALE
# ======================
feature_cols = [
    "value",
    "day",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month",
    "cos_month",
    "sin_week_of_year",
    "cos_week_of_year",
    "is_holiday",
]

scaler = MinMaxScaler()

train_values = scaler.fit_transform(train_df[feature_cols])
test_values = scaler.transform(test_df[feature_cols])

# ======================
# SEQUENCES
# ======================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # target is always "value"
    return np.array(X), np.array(y).reshape(-1, 1)

X_train, y_train = create_sequences(train_values, SEQ_LENGTH)
X_test, y_test = create_sequences(test_values, SEQ_LENGTH)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ======================
# TENSORS
# ======================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

INPUT_SIZE = X_train.shape[2]
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

# ======================
# TRAIN FUNCTION
# ======================
def train_model(model):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")

    return model

# ======================
# EVALUATE
# ======================
def inverse_transform_target(values):
    temp = np.zeros((len(values), len(feature_cols)))
    temp[:, 0] = values.reshape(-1)
    return scaler.inverse_transform(temp)[:, 0].reshape(-1, 1)

def evaluate(model, name):
    model.eval()
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

    print(f"{name} MAE: {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")

    return preds, actuals, mae, rmse

# ======================
# TRAIN ALL
# ======================
rnn = train_model(RNNModel())
lstm = train_model(LSTMModel())
gru = train_model(GRUModel())

# ======================
# EVALUATE ALL
# ======================
rnn_preds, _, rnn_mae, rnn_rmse = evaluate(rnn, "RNN")
lstm_preds, _, lstm_mae, lstm_rmse = evaluate(lstm, "LSTM")
gru_preds, _, gru_mae, gru_rmse = evaluate(gru, "GRU")

# ======================
# SAVE BEST MODEL
# ======================
os.makedirs("../saved_models", exist_ok=True)

results = {
    "RNN": rnn_rmse,
    "LSTM": lstm_rmse,
    "GRU": gru_rmse
}

best_model_name = min(results, key=results.get)
print("Best model:", best_model_name)

if best_model_name == "RNN":
    best_model = rnn
elif best_model_name == "LSTM":
    best_model = lstm
else:
    best_model = gru

torch.save(
    {
        "model_name": best_model_name,
        "state_dict": best_model.state_dict(),
        "feature_cols": feature_cols,
        "seq_length": SEQ_LENGTH,
    },
    "../saved_models/best_model_v2.pt"
)

print("Best model saved to ../saved_models/best_model_v2.pt")