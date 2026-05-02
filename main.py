from preprocessing import build_dataset
from models.lstm import run_lstm
from models.gru import run_gru

FILEPATH = 'data/Scats Data October 2006.xls'

X_train, X_test, y_train, y_test, scalers, coords, agg = build_dataset(FILEPATH)

# LSTM
lstm_predictions, y_test_scaled = run_lstm(X_train, X_test, y_train, y_test, scalers)

# GRU
gru_predictions = run_gru(X_train, X_test, y_train, y_test, scalers)