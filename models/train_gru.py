"""Train the GRU model and save its final test metrics.

Run from project root:
    python3 models/train_gru.py

Outputs in saved_models/:
    gru_model.keras
    flow_scaler.joblib
    model_metrics.csv
    model_training_history.csv
    model_test_predictions.csv
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GRU, Input

def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = Sequential([Input(shape=input_shape), GRU(64, return_sequences=True), Dropout(0.2), GRU(32), Dropout(0.2), Dense(16, activation="relu"), Dense(1)], name="scats_gru")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


from model_utils import (
    inverse_flow,
    prepare_model_data,
    print_split_summary,
    regression_metrics,
    saved_models_dir,
    update_metrics_file,
    update_history_file,
    update_predictions_file,
)

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = prepare_model_data(save_artifacts=True)
    print_split_summary(data)

    model_path = saved_models_dir() / "gru_model.keras"
    model = build_model(data["input_shape"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False,
        callbacks=callbacks, verbose=1,
    )

    best_model = tf.keras.models.load_model(model_path)
    predicted_scaled = best_model.predict(data["X_test"], verbose=0)
    actual_flow = inverse_flow(data["y_test"], data["scaler"])
    predicted_flow = inverse_flow(predicted_scaled, data["scaler"])

    metrics = regression_metrics(actual_flow, predicted_flow)
    metrics_file = update_metrics_file("GRU", metrics)
    history_file = update_history_file("GRU", history.history)
    predictions_file = update_predictions_file(
        "GRU", data["test_meta"], actual_flow, predicted_flow
    )

    print("\nGRU final test metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\nSaved model:", model_path)
    print("Updated metrics file:", metrics_file)
    print("Updated training history file:", history_file)
    print("Updated test predictions file:", predictions_file)


if __name__ == "__main__":
    main()
