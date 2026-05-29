from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "CNN"

INTERVAL_MINUTES = 15
DAY_SECONDS = 24 * 60 * 60

SPEED_LIMIT_KMH = 60.0
CAPACITY_FLOW_PER_HOUR = 1500.0
MIN_SPEED_KMH = 1.0

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
    """Convert HH:MM or HH:MM:SS into seconds from midnight."""
    parts = str(value).strip().split(":")

    if len(parts) not in (2, 3):
        raise ValueError(
            "Departure time must use HH:MM or HH:MM:SS format."
        )

    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as error:
        raise ValueError(
            "Departure time must use HH:MM or HH:MM:SS format."
        ) from error

    if not 0 <= hour <= 23:
        raise ValueError("Departure hour must be between 00 and 23.")

    if not 0 <= minute <= 59:
        raise ValueError("Departure minute must be between 00 and 59.")

    if not 0 <= second <= 59:
        raise ValueError("Departure second must be between 00 and 59.")

    return float(hour * 3600 + minute * 60 + second)


def format_time_of_day(time_seconds: float) -> str:
    """Format seconds from midnight as HH:MM:SS"""
    total_seconds = int(round(time_seconds)) % DAY_SECONDS

    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    second = total_seconds % 60

    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _time_interval(time_seconds: float) -> int:
    """Return the 15-minute interval containing the segment entry time"""
    return int(time_seconds % DAY_SECONDS) // (INTERVAL_MINUTES * 60)


def _interval_label(interval: int) -> str:
    """Return the readable label of one 15-minute traffic interval"""
    start_seconds = interval * INTERVAL_MINUTES * 60
    end_seconds = start_seconds + INTERVAL_MINUTES * 60 - 1

    return (
        f"{format_time_of_day(start_seconds)}"
        f"-{format_time_of_day(end_seconds)}"
    )


def _calculate_curve_speed(
    curve_flow: float,
    congested_branch: bool,
) -> float:
    """Calculate one speed solution from the supplied parabolic curve"""
    discriminant = B**2 - 4 * A * (-curve_flow)

    if discriminant < 0:
        raise ValueError("Cannot convert predicted flow into speed.")

    if congested_branch:
        # Lower-speed root: red / over-capacity branch.
        return (-B + math.sqrt(discriminant)) / (2 * A)

    # Higher-speed root: green / under-capacity branch.
    return (-B - math.sqrt(discriminant)) / (2 * A)


def flow_to_speed(flow_per_hour: float) -> float:
    """
    Under-capacity traffic uses the green branch
    Above-capacity traffic uses the red branch
    """

    if flow_per_hour < 0:
        raise ValueError("Predicted flow cannot be negative.")

    # At low flow, calculated speed would exceed the speed limit.
    if flow_per_hour <= 351:
        return SPEED_LIMIT_KMH

    if flow_per_hour <= CAPACITY_FLOW_PER_HOUR:
        speed_kmh = _calculate_curve_speed(
            curve_flow=flow_per_hour,
            congested_branch=False,
        )

        return min(speed_kmh, SPEED_LIMIT_KMH)

    # The part above capacity indicates increased congestion
    # Mirror the exceeded amount onto the red side of the curve
    overflow = flow_per_hour - CAPACITY_FLOW_PER_HOUR
    congested_curve_flow = max(
        0.0,
        CAPACITY_FLOW_PER_HOUR - overflow,
    )

    if congested_curve_flow == 0:
        return MIN_SPEED_KMH

    speed_kmh = _calculate_curve_speed(
        curve_flow=congested_curve_flow,
        congested_branch=True,
    )

    return max(speed_kmh, MIN_SPEED_KMH)


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
            raise ValueError(
                "final_edges.csv contains invalid distance_km values."
            )

        if (edges["distance_km"] < 0).any():
            raise ValueError(
                "final_edges.csv contains negative distance_km values."
            )

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

        predictions["site_id"] = predictions["site_id"].apply(
            _normalise_site_id
        )

        predictions["target_timestamp"] = pd.to_datetime(
            predictions["target_timestamp"],
            errors="coerce",
        )

        predictions["predicted_total_flow"] = pd.to_numeric(
            predictions["predicted_total_flow"],
            errors="coerce",
        )

        if predictions["target_timestamp"].isna().any():
            raise ValueError(
                "Prediction data contains invalid timestamps."
            )

        if predictions["predicted_total_flow"].isna().any():
            raise ValueError(
                "Prediction data contains invalid flow values."
            )

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
        """Return SCATS sites available in the route graph"""
        return set(self.edges["from_site"]) | set(self.edges["to_site"])

    def neighbours(self, site_id: str) -> list[str]:
        """Return directly connected next SCATS sites"""
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
        """Calculate dynamic travel cost and log values for one segment"""
        from_site = _normalise_site_id(from_site)
        to_site = _normalise_site_id(to_site)

        edge = self.edges[
            (self.edges["from_site"] == from_site)
            & (self.edges["to_site"] == to_site)
        ]

        if edge.empty:
            raise ValueError(
                f"No edge found from {from_site} to {to_site}."
            )

        interval = _time_interval(entry_time_seconds)
        flow_key = (from_site, interval)

        if flow_key not in self.flow_lookup:
            raise ValueError(
                f"No predicted flow found for SCATS site {from_site} "
                f"at interval {_interval_label(interval)}."
            )

        predicted_flow_15_min = self.flow_lookup[flow_key]

        # The model predicts 15-minute flow; the conversion uses hourly flow
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
        """Return only the segment cost used by the route algorithm"""
        detail = self.get_segment_detail(
            from_site=from_site,
            to_site=to_site,
            entry_time_seconds=entry_time_seconds,
        )

        return detail["segment_time_minutes"]