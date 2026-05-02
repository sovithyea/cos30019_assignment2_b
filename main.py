from preprocessing import build_dataset
from models.lstm import run_lstm
from models.gru import run_gru

FILEPATH = 'data/Scats Data October 2006.xls'

X_train, X_test, y_train, y_test, scalers, coords, agg = build_dataset(FILEPATH)

# LSTM
lstm_mae, lstm_rmse = run_lstm(X_train, X_test, y_train, y_test, scalers)

# GRU
gru_mae, gru_rmse = run_gru(X_train, X_test, y_train, y_test, scalers)

print("\nTraffic Data")
print("Max traffic:", round(agg["flow"].max(), 2))
print("Average traffic:", round(agg["flow"].mean(), 2))

print("\nModel Results")
print("LSTM -> MAE:", round(lstm_mae, 5), "| RMSE:", round(lstm_rmse, 5))
print("GRU  -> MAE:", round(gru_mae, 5), "| RMSE:", round(gru_rmse, 5))
print("\n")