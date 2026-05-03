import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_lstm(X_train, X_test, y_train, y_test, scalers):
    model = Sequential()
    model.add(Input(shape=(12, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )

    predictions_scaled = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_scaled))

    print("LSTM MAE:", mae)
    print("LSTM RMSE:", rmse)

    return model, predictions_scaled, history, mae, rmse