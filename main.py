import os
import pickle
from preprocessing import build_dataset
from models.lstm import run_lstm
from models.gru import run_gru
from evaluate import evaluate_all

FILEPATH = 'data/Scats Data October 2006.xls'

X_train, X_test, y_train, y_test, scalers, coords, agg = build_dataset(FILEPATH)

# LSTM
lstm_model, lstm_preds, lstm_history, lstm_mae, lstm_rmse = run_lstm(X_train, X_test, y_train, y_test, scalers)

# GRU
gru_model, gru_preds, gru_history, gru_mae, gru_rmse = run_gru(X_train, X_test, y_train, y_test, scalers)

print("\nTraffic Data")
print("Max traffic:", round(agg["flow"].max(), 2))
print("Average traffic:", round(agg["flow"].mean(), 2))

print("\nModel Results")
print("LSTM -> MAE:", round(lstm_mae, 5), "| RMSE:", round(lstm_rmse, 5))
print("GRU  -> MAE:", round(gru_mae, 5), "| RMSE:", round(gru_rmse, 5))

# evaluate and compare all models
results = [
    {'name': 'LSTM', 'predictions': lstm_preds, 'history': lstm_history, 'mae': lstm_mae, 'rmse': lstm_rmse},
    {'name': 'GRU', 'predictions': gru_preds, 'history': gru_history, 'mae': gru_mae, 'rmse': gru_rmse},
]

evaluate_all(results, y_test, scalers)

# save scalers so they can be loaded later without reprocessing
os.makedirs('saved_models', exist_ok=True)
with open('saved_models/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
print('Saved saved_models/scalers.pkl')