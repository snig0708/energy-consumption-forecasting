import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os


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

# ======================
# VISUALIZE
# ======================
plt.figure(figsize=(15,5))
plt.plot(df["datetime"], df["value"])
plt.title("Energy Consumption")
plt.show()

# ======================
# SPLIT
# ======================
train_size = int(len(df) * 0.8)

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# ======================
# SCALE
# ======================
scaler = MinMaxScaler()

train_values = scaler.fit_transform(train_df[["value"]])
test_values = scaler.transform(test_df[["value"]])

# ======================
# SEQUENCES
# ======================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_values, SEQ_LENGTH)
X_test, y_test = create_sequences(test_values, SEQ_LENGTH)

# ======================
# TENSORS
# ======================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ======================
# MODELS
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
# TRAIN FUNCTION
# ======================
def train_model(model):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}")

    return model


# ======================
# EVALUATE
# ======================
def evaluate(model, name):
    model.eval()
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
    torch.save(rnn.state_dict(), "../saved_models/best_model.pt")
elif best_model_name == "LSTM":
    torch.save(lstm.state_dict(), "../saved_models/best_model.pt")
else:
    torch.save(gru.state_dict(), "../saved_models/best_model.pt")


print("Best model saved to ../saved_models/best_model.pt")


