from preprocessing import build_dataset
from models.lstm import run_lstm



FILEPATH = 'data/Scats Data October 2006.xls'

X_train, X_test, y_train, y_test, scalers, coords, agg = build_dataset(FILEPATH)

# -- Run LSTM model --
predictions_scaled, y_test = run_lstm(X_train, X_test, y_train, y_test, scalers)