import numpy as np
import pandas as pd

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
DATA_PATH = "../data/PJME_hourly.csv"
MODEL_PATH = "../saved_models/best_model.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ======================
# MODELS (ALL THREE)
# ======================
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32, device=x.device)
        c0 = torch.zeros(2, x.size(0), 32, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 32, device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


# ======================
# DATA PREP (same as train)
# ======================
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "Datetime": "datetime",
    "PJME_MW": "value"
})

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

scaler = MinMaxScaler()
train_values = scaler.fit_transform(train_df[["value"]])
test_values = scaler.transform(test_df[["value"]])


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


X_test, y_test = create_sequences(test_values, SEQ_LENGTH)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ======================
# LOAD MODEL
# ======================

# ⚡ IMPORTANT: change this manually based on training output
BEST_MODEL = "RNN"   # <-- change if needed

if BEST_MODEL == "RNN":
    model = RNNModel()
elif BEST_MODEL == "LSTM":
    model = LSTMModel()
else:
    model = GRUModel()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ======================
# EVALUATE
# ======================
preds, actuals = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)

        preds.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

preds = scaler.inverse_transform(preds)
actuals = scaler.inverse_transform(actuals)

mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")