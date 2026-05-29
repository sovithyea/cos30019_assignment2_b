"""
RESEARCH - Mixed precision training for the selected GRU model.

This experiment applies mixed precision training to the GRU architecture
while keeping the original GRU hyperparameters unchanged. This allows a
controlled comparison against the original FP32 GRU baseline.

Run from project root:
    python3 models/train_gru_research.py

Outputs in saved_models/:
    gru_mixed_controlled_model.keras
    flow_scaler.joblib
    model_metrics.csv
    model_training_history.csv
    model_test_predictions.csv
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GRU, Input

from model_utils import (
    inverse_flow,
    prepare_model_data,
    print_split_summary,
    regression_metrics,
    saved_models_dir,
    update_history_file,
    update_metrics_file,
    update_predictions_file,
)


# Controlled research configuration

MODEL_LABEL = "GRU_MIXED_CONTROLLED"

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Mixed precision is the only research technique being applied.
mixed_precision.set_global_policy("mixed_float16")


# Model definition

def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=input_shape),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, dtype="float32"),
        ],
        name="scats_gru_mixed_controlled",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    return model


# Training and evaluation

def main() -> None:
    # Train and evaluate the controlled mixed-precision GRU experiment
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    data = prepare_model_data(save_artifacts=True)
    print_split_summary(data)

    model_path = saved_models_dir() / "gru_mixed_controlled_model.keras"

    model = build_model(data["input_shape"])
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    training_start = time.perf_counter()

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    training_time_seconds = time.perf_counter() - training_start

    # Load the best saved checkpoint before evaluating on the test set
    # matching the evaluation procedure used by CNN, GRU and LSTM.
    best_model = tf.keras.models.load_model(model_path)

    predicted_scaled = best_model.predict(
        data["X_test"],
        verbose=0,
    )

    actual_flow = inverse_flow(
        data["y_test"],
        data["scaler"],
    )

    predicted_flow = inverse_flow(
        predicted_scaled,
        data["scaler"],
    )

    metrics = regression_metrics(
        actual_flow,
        predicted_flow,
    )

    metrics_file = update_metrics_file(
        MODEL_LABEL,
        metrics,
    )

    history_file = update_history_file(
        MODEL_LABEL,
        history.history,
    )

    predictions_file = update_predictions_file(
        MODEL_LABEL,
        data["test_meta"],
        actual_flow,
        predicted_flow,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history.history["val_loss"]))
    epochs_run = len(history.history["loss"])

    print("\nControlled Mixed-Precision GRU Results")
    print("-" * 45)
    print(f"Model label:           {MODEL_LABEL}")
    print(f"Precision policy:      {mixed_precision.global_policy().name}")
    print(f"Batch size:            {BATCH_SIZE}")
    print(f"Learning rate:         {LEARNING_RATE}")
    print(f"Epochs run:            {epochs_run}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation loss:  {best_val_loss:.6f}")
    print(f"Training time:         {training_time_seconds:.2f} seconds")

    print("\nFinal test metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\nSaved model:", model_path)
    print("Updated metrics file:", metrics_file)
    print("Updated training history file:", history_file)
    print("Updated test predictions file:", predictions_file)


if __name__ == "__main__":
    main()