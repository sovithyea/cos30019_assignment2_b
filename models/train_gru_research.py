"""RESEARCH - Improving GRU model efficiency and accuracy 

Train the GRU model and save its final test metrics.

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
from tensorflow.keras import Sequential, mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GRU, Input

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

#GRU Model config
SEED = 42
EPOCHS = 60
BATCH_SIZE = 256          #changed from 64 to 256
LR = 0.001 * (BATCH_SIZE / 64)   # = 0.004 learning rate. Implemented learning rate change to account for larger batch size, 
                                 #helps control the steps the models take


#suggestion from AI assistance, but further research done on a conferece paper on mixed precision training
#this essentially speeds up computation by using float16 and keeps a master copy of weights in float32 for stabilitiy 
mixed_precision.set_global_policy("mixed_float16") 


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """GRU model.

    Notes
    -----
    * ``reset_after=True`` (Keras default) is stated explicitly because it is
      required to enable the cuDNN-optimised GRU kernel on GPU (2–5× faster).
    * The final Dense layer uses ``dtype="float32"`` so the output is kept in
      full precision even when mixed-precision is active.
    """
    model = Sequential(
        [
            #model settings
            Input(shape=input_shape),
            GRU(64, return_sequences=True, reset_after=True),
            Dropout(0.2),
            GRU(32, reset_after=True),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, dtype="float32"),   #addition now mixed precision training included to keep results in float32
        ],
        name="scats_gru",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss="mse", metrics=["mae"],
    )
    return model


#replaces model.fit to implement the tf.data prefetch and cache tasks 
def make_dataset(X, y, batch_size, shuffle=False,) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    return (
        ds.batch(batch_size)
          .cache()                      #caches data in RAM for faster analysis
          .prefetch(tf.data.AUTOTUNE)   #prefetch allows GPU to train model while CPI is preping dataset
    )


def main() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = prepare_model_data(save_artifacts=True)
    print_split_summary(data)

    model_path = saved_models_dir() / "gru_model.keras"
    model = build_model(data["input_shape"])
    model.summary()

    # t.data call
    train_ds = make_dataset(
        data["X_train"], 
        data["y_train"], 
        BATCH_SIZE, shuffle=False
    )
    val_ds   = make_dataset(
        data["X_val"],   
        data["y_val"],   
        BATCH_SIZE
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    predicted_scaled = model.predict(data["X_test"], verbose=0)
    actual_flow = inverse_flow(data["y_test"],        data["scaler"])
    predicted_flow = inverse_flow(predicted_scaled,      data["scaler"])

    metrics = regression_metrics(actual_flow, predicted_flow)
    metrics_file = update_metrics_file("GRU", metrics)
    history_file = update_history_file("GRU", history.history)
    predictions_file = update_predictions_file(
        "GRU", data["test_meta"], actual_flow, predicted_flow
    )

    print("\nGRU final test metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("\nSaved model:", model_path)
    print("Updated metrics:", metrics_file)
    print("Updated history:", history_file)
    print("Updated predictions:", predictions_file)


if __name__ == "__main__":
    main()