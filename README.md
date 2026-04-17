# PJM Energy Consumption Forecasting using RNN, LSTM, and GRU

This project forecasts PJM hourly energy consumption using three deep learning sequence models:

- RNN
- LSTM
- GRU

## Project workflow

- Load and clean PJM hourly data
- Sort data chronologically
- Split into train and test sets by time
- Scale the features
- Create fixed-length input sequences
- Train RNN, LSTM, and GRU models
- Compare MAE and RMSE
- Save plots and the best-performing model

## Version 2: Calendar + Cyclical Features

This version extends the baseline by adding time-based engineered features:
- day of month
- cyclical hour encoding
- cyclical day-of-week encoding
- cyclical month encoding
- cyclical week-of-year encoding
- US holiday flag

These features help the model capture recurring seasonal and calendar-driven patterns in electricity demand.

## Project structure

```text
pjm-energy-forecasting/
├── data/
├── src/
├── results/
├── saved_models/
├── requirements.txt
├── .gitignore
└── README.md


