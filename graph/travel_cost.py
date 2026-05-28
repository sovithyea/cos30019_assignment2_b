from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "LSTM"
INTERVAL_MINUTES = 15
DAY_SECONDS = 24 * 60 * 60

SPEED_LIMIT_KMH = 60.0
CAPACITY_FLOW_PER_HOUR = 1500.0

# Supplied flow-speed conversion formula:
# flow = A * speed^2 + B * speed
A = -1.4648375
B = 93.75


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalise_site_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def parse_departure_time(value: str) -> float:
    # Parse to HH:MM or HH:MM:SS
    parts = value.strip().split(":")

    if len(parts) not in (2, 3):
        raise ValueError("Departure time must use HH:MM or HH:MM:SS format.")

    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as error:
        raise ValueError(
            "Departure time must use HH:MM or HH:MM:SS format."
        ) from error

    if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ValueError("Departure time is invalid.")

    return float(hour * 3600 + minute * 60 + second)


def format_time_of_day(time_seconds: float) -> str:
    total_seconds = int(round(time_seconds)) % DAY_SECONDS

    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60

    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _time_interval(time_seconds: float) -> int:
    # Use the 15-minute interval containing the segment entry time
    return int(time_seconds % DAY_SECONDS) // (INTERVAL_MINUTES * 60)


def _interval_label(interval: int) -> str:
    start_seconds = interval * INTERVAL_MINUTES * 60
    end_seconds = start_seconds + INTERVAL_MINUTES * 60 - 1

    return (
        f"{format_time_of_day(start_seconds)}"
        f"-{format_time_of_day(end_seconds)}"
    )


def flow_to_speed(flow_per_hour: float) -> float:
    # Estimate speed from predicted hourly flow
    if flow_per_hour < 0:
        raise ValueError("Predicted flow cannot be negative.")

    if flow_per_hour <= 351:
        return SPEED_LIMIT_KMH

    flow_per_hour = min(flow_per_hour, CAPACITY_FLOW_PER_HOUR)

    discriminant = B**2 - 4 * A * (-flow_per_hour)

    if discriminant < 0:
        raise ValueError("Cannot convert predicted flow into speed.")

    # Higher-speed root = under-capacity branch
    speed_kmh = (-B - math.sqrt(discriminant)) / (2 * A)

    return min(speed_kmh, SPEED_LIMIT_KMH)


class TravelTimeEstimator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        root: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.root = root if root is not None else _project_root()

        self.edges = self._load_edges()
        self.flow_lookup = self._load_predicted_flow()

    def _load_edges(self) -> pd.DataFrame:
        edge_file = self.root / "data" / "route_map" / "final_edges.csv"

        if not edge_file.exists():
            raise FileNotFoundError(f"Cannot find {edge_file}.")

        edges = pd.read_csv(edge_file)

        required = {"from_site", "to_site", "distance_km"}
        missing = required.difference(edges.columns)

        if missing:
            raise ValueError(
                f"final_edges.csv is missing required columns: {sorted(missing)}"
            )

        edges["from_site"] = edges["from_site"].apply(_normalise_site_id)
        edges["to_site"] = edges["to_site"].apply(_normalise_site_id)
        edges["distance_km"] = pd.to_numeric(
            edges["distance_km"],
            errors="coerce",
        )

        if edges["distance_km"].isna().any():
            raise ValueError("final_edges.csv contains invalid distance_km values.")

        return (
            edges
            .sort_values("distance_km")
            .drop_duplicates(["from_site", "to_site"])
            .reset_index(drop=True)
        )

    def _load_predicted_flow(self) -> dict[tuple[str, int], float]:
        prediction_file = (
            self.root
            / "saved_models"
            / "model_test_predictions.csv"
        )

        if not prediction_file.exists():
            raise FileNotFoundError(f"Cannot find {prediction_file}.")

        predictions = pd.read_csv(prediction_file)

        required = {
            "Model",
            "site_id",
            "target_timestamp",
            "predicted_total_flow",
        }
        missing = required.difference(predictions.columns)

        if missing:
            raise ValueError(
                "model_test_predictions.csv is missing required columns: "
                f"{sorted(missing)}"
            )

        predictions = predictions[
            predictions["Model"].astype(str).str.upper()
            == self.model_name.upper()
        ].copy()

        if predictions.empty:
            raise ValueError(
                f"No prediction data found for model '{self.model_name}'."
            )

        predictions["site_id"] = predictions["site_id"].apply(_normalise_site_id)
        predictions["target_timestamp"] = pd.to_datetime(
            predictions["target_timestamp"],
            errors="coerce",
        )
        predictions["predicted_total_flow"] = pd.to_numeric(
            predictions["predicted_total_flow"],
            errors="coerce",
        )

        if predictions["target_timestamp"].isna().any():
            raise ValueError("Prediction data contains invalid timestamps.")

        if predictions["predicted_total_flow"].isna().any():
            raise ValueError("Prediction data contains invalid flow values.")

        predictions["interval"] = (
            predictions["target_timestamp"].dt.hour * 4
            + predictions["target_timestamp"].dt.minute // INTERVAL_MINUTES
        )

        average_flow = (
            predictions
            .groupby(["site_id", "interval"], as_index=False)[
                "predicted_total_flow"
            ]
            .mean()
        )

        return {
            (row.site_id, int(row.interval)): float(row.predicted_total_flow)
            for row in average_flow.itertuples(index=False)
        }

    def valid_nodes(self) -> set[str]:
        return set(self.edges["from_site"]) | set(self.edges["to_site"])

    def neighbours(self, site_id: str) -> list[str]:
        site_id = _normalise_site_id(site_id)

        return self.edges.loc[
            self.edges["from_site"] == site_id,
            "to_site",
        ].tolist()

    def get_segment_detail(
        self,
        from_site: str,
        to_site: str,
        entry_time_seconds: float,
    ) -> dict:
        from_site = _normalise_site_id(from_site)
        to_site = _normalise_site_id(to_site)

        edge = self.edges[
            (self.edges["from_site"] == from_site)
            & (self.edges["to_site"] == to_site)
        ]

        if edge.empty:
            raise ValueError(f"No edge found from {from_site} to {to_site}.")

        interval = _time_interval(entry_time_seconds)
        flow_key = (from_site, interval)

        if flow_key not in self.flow_lookup:
            raise ValueError(
                f"No predicted flow found for SCATS site {from_site} "
                f"at interval {_interval_label(interval)}."
            )

        predicted_flow_15_min = self.flow_lookup[flow_key]

        # Internal values used only to calculate segment travel time
        flow_per_hour = predicted_flow_15_min * 4
        speed_kmh = flow_to_speed(flow_per_hour)
        distance_km = float(edge.iloc[0]["distance_km"])

        segment_time_minutes = (distance_km / speed_kmh) * 60
        

        arrival_time_seconds = entry_time_seconds + segment_time_minutes * 60

        return {
            "from_site": from_site,
            "to_site": to_site,
            "entry_time": format_time_of_day(entry_time_seconds),
            "traffic_interval": _interval_label(interval),
            "predicted_flow_15_min": predicted_flow_15_min,
            "predicted_flow_per_hour": flow_per_hour,
            "speed_kmh": speed_kmh,
            "distance_km": distance_km,
            "segment_time_minutes": segment_time_minutes,
            "arrival_time": format_time_of_day(arrival_time_seconds),
        }

    def edge_travel_time(
        self,
        from_site: str,
        to_site: str,
        entry_time_seconds: float,
    ) -> float:
        detail = self.get_segment_detail(
            from_site=from_site,
            to_site=to_site,
            entry_time_seconds=entry_time_seconds,
        )

        return detail["segment_time_minutes"]