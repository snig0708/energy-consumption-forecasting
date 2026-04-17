# PJM Energy Consumption Forecasting with RNN, LSTM, and GRU

This project forecasts hourly energy consumption using deep learning sequence models built in PyTorch.

The models implemented are:
- RNN
- LSTM
- GRU

## Project Overview

The goal of this project is to compare recurrent neural network architectures on PJM hourly energy consumption data and evaluate which model performs best for forecasting.

The workflow includes:
- loading and preprocessing time series data
- sorting the data chronologically
- splitting train and test sets by time
- scaling the target variable
- creating sequential input windows
- training RNN, LSTM, and GRU models
- comparing performance using MAE and RMSE

## Dataset

This project uses the PJM hourly energy consumption dataset.

Expected file location:

```text
data/PJME_hourly.csv