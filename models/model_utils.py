"""
model_utils.py
COS30019 Assignment 2B - Shared model data pipeline.

Place this file in:
    cos30019_assignment2_b/models/model_utils.py

Expected processed input:
    cos30019_assignment2_b/data/processed/traffic_direction_timeseries.csv

Important design decision:
- The processed CSV remains direction-level for EDA/visualisation.
- The model pipeline aggregates directional flow into total_flow for each
  SCATS site and timestamp because the routing scenario operates on the
  40 SCATS sites as nodes.
- Train/validation/test splitting is time-based:
    Train:      2006-10-01 to 2006-10-20
    Validation: 2006-10-21 to 2006-10-25
    Test:       2006-10-26 to 2006-10-31
- The scaler is fit on the training set only to avoid data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Configuration
# ============================================================

LOOKBACK = 12          # 12 x 15 minutes = 3 hours of history
HORIZON = 1            # predict the next 15-minute interval
EXPECTED_INTERVAL = pd.Timedelta(minutes=15)

TRAIN_START = pd.Timestamp("2006-10-01 00:00:00")
TRAIN_END = pd.Timestamp("2006-10-20 23:45:00")
VAL_START = pd.Timestamp("2006-10-21 00:00:00")
VAL_END = pd.Timestamp("2006-10-25 23:45:00")
TEST_START = pd.Timestamp("2006-10-26 00:00:00")
TEST_END = pd.Timestamp("2006-10-31 23:45:00")

FEATURE_COLUMNS = [
    "flow_scaled",
    "interval_sin",
    "interval_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
TARGET_COLUMN = "flow_scaled"
ACTUAL_FLOW_COLUMN = "total_flow"


# ============================================================
# Paths
# ============================================================

def project_root() -> Path:
    """Return the project root assuming this file is inside models/."""
    return Path(__file__).resolve().parent.parent


def processed_data_file() -> Path:
    return project_root() / "data" / "processed" / "traffic_direction_timeseries.csv"


def saved_models_dir() -> Path:
    folder = project_root() / "saved_models"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ============================================================
# Load and prepare modelling dataframe
# ============================================================

def load_direction_level_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the clean direction-level dataset produced by preprocessing_data.ipynb."""
    path = Path(csv_path) if csv_path is not None else processed_data_file()

    if not path.exists():
        raise FileNotFoundError(
            f"Processed traffic file not found: {path}\n"
            "Run data/preprocessing_data.ipynb first."
        )

    df = pd.read_csv(path)

    required_columns = {"site_id", "location", "timestamp", "flow"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in processed data: {sorted(missing_columns)}")

    df = df.copy()
    df["site_id"] = (
        df["site_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(4)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")

    invalid_rows = df[df["timestamp"].isna() | df["flow"].isna()]
    if not invalid_rows.empty:
        raise ValueError(
            f"Processed dataset contains {len(invalid_rows)} rows with invalid timestamp/flow."
        )

    return df.sort_values(["site_id", "location", "timestamp"]).reset_index(drop=True)


def aggregate_to_site_level(direction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate directional readings into site-level total flow.

    Result:
        one row = one SCATS site at one 15-minute timestamp.
    """
    site_df = (
        direction_df
        .groupby(["site_id", "timestamp"], as_index=False)
        .agg(
            total_flow=("flow", "sum"),
            direction_count=("location", "nunique"),
        )
        .sort_values(["site_id", "timestamp"])
        .reset_index(drop=True)
    )

    site_df["date_only"] = site_df["timestamp"].dt.date
    site_df["interval"] = (
        site_df["timestamp"].dt.hour * 4
        + site_df["timestamp"].dt.minute // 15
    )
    site_df["day_of_week"] = site_df["timestamp"].dt.dayofweek
    site_df["is_weekend"] = site_df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical time features retain the repeating daily/weekly pattern.
    site_df["interval_sin"] = np.sin(2 * np.pi * site_df["interval"] / 96)
    site_df["interval_cos"] = np.cos(2 * np.pi * site_df["interval"] / 96)
    site_df["dow_sin"] = np.sin(2 * np.pi * site_df["day_of_week"] / 7)
    site_df["dow_cos"] = np.cos(2 * np.pi * site_df["day_of_week"] / 7)

    return site_df


# ============================================================
# Time-based split and scaling
# ============================================================

def split_by_time(site_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split all sites by future time periods, never randomly."""
    train_df = site_df[
        (site_df["timestamp"] >= TRAIN_START) &
        (site_df["timestamp"] <= TRAIN_END)
    ].copy()

    val_df = site_df[
        (site_df["timestamp"] >= VAL_START) &
        (site_df["timestamp"] <= VAL_END)
    ].copy()

    test_df = site_df[
        (site_df["timestamp"] >= TEST_START) &
        (site_df["timestamp"] <= TEST_END)
    ].copy()

    for split_name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df),
    ]:
        if split_df.empty:
            raise ValueError(f"{split_name} set is empty. Check timestamp parsing and split dates.")

    return train_df, val_df, test_df


def scale_flow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Fit the target-flow scaler using training data only."""
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled["flow_scaled"] = scaler.fit_transform(train_scaled[["total_flow"]])
    val_scaled["flow_scaled"] = scaler.transform(val_scaled[["total_flow"]])
    test_scaled["flow_scaled"] = scaler.transform(test_scaled[["total_flow"]])

    return train_scaled, val_scaled, test_scaled, scaler


# ============================================================
# Sequence generation
# ============================================================

def create_sequences(
    split_df: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create sequences separately for each site.

    A sequence is kept only inside a continuous 15-minute run. This avoids
    crossing missing dates/intervals while remaining efficient for this dataset.
    """
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    metadata: list[dict[str, Any]] = []

    for site_id, group in split_df.groupby("site_id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)

        # Split a site's time series whenever there is a gap larger/smaller
        # than one expected 15-minute interval.
        run_id = group["timestamp"].diff().ne(EXPECTED_INTERVAL).cumsum()

        for _, run in group.groupby(run_id, sort=False):
            run = run.reset_index(drop=True)

            if len(run) < lookback + horizon:
                continue

            features = run[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            targets = run[TARGET_COLUMN].to_numpy(dtype=np.float32)
            timestamps = run["timestamp"].to_numpy()

            for target_start in range(lookback, len(run) - horizon + 1):
                target_index = target_start + horizon - 1
                window_start = target_start - lookback

                x_values.append(features[window_start:target_start])
                y_values.append(targets[target_index])
                metadata.append(
                    {
                        "site_id": site_id,
                        "target_timestamp": pd.Timestamp(timestamps[target_index]),
                        "actual_total_flow": float(run.loc[target_index, "total_flow"]),
                    }
                )

    if not x_values:
        raise ValueError("No valid sequences were created. Check the split and interval continuity.")

    x_array = np.asarray(x_values, dtype=np.float32)
    y_array = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    metadata_df = pd.DataFrame(metadata)

    return x_array, y_array, metadata_df


# ============================================================
# Full shared pipeline
# ============================================================

def prepare_model_data(
    csv_path: str | Path | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Run the shared pipeline used by all three models and evaluation notebook.
    """
    direction_df = load_direction_level_data(csv_path)
    site_df = aggregate_to_site_level(direction_df)
    train_df, val_df, test_df = split_by_time(site_df)
    train_df, val_df, test_df, scaler = scale_flow(train_df, val_df, test_df)

    x_train, y_train, train_meta = create_sequences(train_df)
    x_val, y_val, val_meta = create_sequences(val_df)
    x_test, y_test, test_meta = create_sequences(test_df)

    data = {
        "direction_df": direction_df,
        "site_df": site_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
        "X_test": x_test,
        "y_test": y_test,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "test_meta": test_meta,
        "scaler": scaler,
        "input_shape": (x_train.shape[1], x_train.shape[2]),
    }

    if save_artifacts:
        artifact_dir = saved_models_dir()
        joblib.dump(scaler, artifact_dir / "flow_scaler.joblib")


    return data


# ============================================================
# Evaluation helpers
# ============================================================

def inverse_flow(scaled_values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    values = np.asarray(scaled_values).reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the five final regression metrics used for model comparison."""
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    total_actual = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual != 0 else np.nan
    )

    non_zero = actual != 0
    mape = (
        np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        if non_zero.any() else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "WAPE_percent": float(wape),
        "MAPE_percent": float(mape),
    }


def update_metrics_file(model_name: str, metrics: dict[str, float]) -> Path:
    """Add or replace one model's test metrics in a shared CSV output file."""
    output_file = saved_models_dir() / "model_metrics.csv"
    new_row = pd.DataFrame([{"Model": model_name, **metrics}])

    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing = existing[existing["Model"] != model_name]
        metrics_df = pd.concat([existing, new_row], ignore_index=True)
    else:
        metrics_df = new_row

    model_order = {"LSTM": 0, "GRU": 1, "CNN": 2}
    metrics_df["sort_order"] = metrics_df["Model"].map(model_order).fillna(99)
    metrics_df = (
        metrics_df.sort_values("sort_order")
        .drop(columns="sort_order")
        .reset_index(drop=True)
    )
    metrics_df.to_csv(output_file, index=False)
    return output_file


def build_prediction_frame(
    metadata: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    predictions = metadata.copy()
    predictions["actual_total_flow"] = np.asarray(actual).ravel()
    predictions["predicted_total_flow"] = np.asarray(predicted).ravel()
    predictions["absolute_error"] = np.abs(
        predictions["actual_total_flow"] - predictions["predicted_total_flow"]
    )
    return predictions


def update_history_file(model_name: str, history: dict[str, list[float]]) -> Path:
    """
    Add or replace one model's epoch-level training/validation history in:
        saved_models/model_training_history.csv

    This file is used by evaluate.ipynb to draw loss curves without
    retraining the model.
    """
    output_file = saved_models_dir() / "model_training_history.csv"

    history_df = pd.DataFrame(history).copy()
    history_df.insert(0, "Epoch", np.arange(1, len(history_df) + 1))
    history_df.insert(0, "Model", model_name)

    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing = existing[existing["Model"] != model_name]
        combined = pd.concat([existing, history_df], ignore_index=True)
    else:
        combined = history_df

    model_order = {"LSTM": 0, "GRU": 1, "CNN": 2}
    combined["sort_order"] = combined["Model"].map(model_order).fillna(99)
    combined = (
        combined.sort_values(["sort_order", "Epoch"])
        .drop(columns="sort_order")
        .reset_index(drop=True)
    )
    combined.to_csv(output_file, index=False)
    return output_file


def update_predictions_file(
    model_name: str,
    metadata: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Path:
    """
    Add or replace one model's final test predictions in:
        saved_models/model_test_predictions.csv

    This file is used by evaluate.ipynb to visualise actual versus predicted
    total traffic flow without running model prediction again.
    """
    output_file = saved_models_dir() / "model_test_predictions.csv"

    predictions = build_prediction_frame(metadata, actual, predicted)
    predictions.insert(0, "Model", model_name)

    if output_file.exists():
        existing = pd.read_csv(output_file)
        existing = existing[existing["Model"] != model_name]
        combined = pd.concat([existing, predictions], ignore_index=True)
    else:
        combined = predictions

    combined["target_timestamp"] = pd.to_datetime(combined["target_timestamp"])
    model_order = {"LSTM": 0, "GRU": 1, "CNN": 2}
    combined["sort_order"] = combined["Model"].map(model_order).fillna(99)
    combined = (
        combined.sort_values(["sort_order", "site_id", "target_timestamp"])
        .drop(columns="sort_order")
        .reset_index(drop=True)
    )
    combined.to_csv(output_file, index=False)
    return output_file


def print_split_summary(data: dict[str, Any]) -> None:
    print("========== SITE-LEVEL TIME-BASED SPLIT ==========")
    print("Direction-level rows loaded:", len(data["direction_df"]))
    print("Unique SCATS sites:", data["direction_df"]["site_id"].nunique())
    print("Site-level rows after aggregation:", len(data["site_df"]))

    for name, df, x_values in [
        ("Training", data["train_df"], data["X_train"]),
        ("Validation", data["val_df"], data["X_val"]),
        ("Testing", data["test_df"], data["X_test"]),
    ]:
        print(f"\n{name} set:")
        print("  Period:", df["timestamp"].min(), "to", df["timestamp"].max())
        print("  SCATS sites:", df["site_id"].nunique())
        print("  Site/timestamp rows:", len(df))
        print("  Sequences:", len(x_values))
        print("  Total flow range:", df["total_flow"].min(), "to", df["total_flow"].max())


if __name__ == "__main__":
    prepared = prepare_model_data(save_artifacts=True)
    print_split_summary(prepared)
    print("\nInput shape used by neural networks:", prepared["input_shape"])
    print("Saved scaler to:", saved_models_dir() / "flow_scaler.joblib")
