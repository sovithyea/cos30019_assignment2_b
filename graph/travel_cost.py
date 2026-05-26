from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "LSTM"
INTERVAL_MINUTES = 15
INTERSECTION_DELAY_MINUTES = 0.5


def _project_root() -> Path:
    """Expected location: project_root/graph/travel_cost.py."""
    return Path(__file__).resolve().parent.parent


def _normalise_site_id(value: object) -> str:
    """Normalise SCATS IDs such as 970, 970.0 and '0970'."""
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def parse_departure_time(value: str | datetime) -> datetime:
    """
    Parse HH:MM or HH:MM:SS.

    A fixed reference date is used because the system uses time-of-day traffic
    patterns rather than requiring the user to select a particular date.
    """
    if isinstance(value, datetime):
        return value

    text = value.strip()

    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text, time_format)

            return datetime(
                year=2006,
                month=10,
                day=26,
                hour=parsed.hour,
                minute=parsed.minute,
                second=parsed.second,
            )

        except ValueError:
            continue

    raise ValueError(
        "Departure time must use HH:MM or HH:MM:SS format, e.g. 08:00."
    )


def _time_interval(value: datetime) -> int:
    """
    Select the current 15-minute traffic interval.

    Example:
    08:03:25 -> interval starting at 08:00
    08:15:00 -> interval starting at 08:15
    """
    total_seconds = (
        value.hour * 3600
        + value.minute * 60
        + value.second
    )

    return total_seconds // (INTERVAL_MINUTES * 60)


def _interval_label(interval: int) -> str:
    """Return a readable interval label such as 08:00-08:14:59."""
    start_minutes = interval * INTERVAL_MINUTES
    start_hour = start_minutes // 60
    start_minute = start_minutes % 60

    end_minutes = start_minutes + INTERVAL_MINUTES - 1
    end_hour = end_minutes // 60
    end_minute = end_minutes % 60

    return (
        f"{start_hour:02d}:{start_minute:02d}:00"
        f"-{end_hour:02d}:{end_minute:02d}:59"
    )


def flow_to_speed(flow_per_hour: float) -> float:
    """Convert predicted hourly traffic flow into estimated speed in km/h."""
    a = -1.46484375
    b = 93.75
    speed_limit = 60.0

    if flow_per_hour <= 351:
        return speed_limit

    discriminant = b**2 - 4 * a * (-flow_per_hour)

    if discriminant < 0:
        return 32.0

    speed_1 = (-b + math.sqrt(discriminant)) / (2 * a)
    speed_2 = (-b - math.sqrt(discriminant)) / (2 * a)

    possible_speeds = [
        speed
        for speed in (speed_1, speed_2)
        if speed > 0
    ]

    if not possible_speeds:
        return 32.0

    return min(max(possible_speeds), speed_limit)


class TravelTimeEstimator:
    """
    Calculate edge travel time using predicted LSTM flow at the entry time.

    Because the user enters time only, not date, predicted flow for the same
    15-minute interval is averaged across the available testing days.
    """

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
        """Load directed SCATS road connections."""
        edge_file = self.root / "data" / "route_map" / "final_edges.csv"

        if not edge_file.exists():
            raise FileNotFoundError(
                f"Cannot find {edge_file}. Run graph/build_route_map.py first."
            )

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

        # If duplicate directed links exist, keep the shortest road segment.
        edges = (
            edges
            .sort_values("distance_km")
            .drop_duplicates(["from_site", "to_site"])
            .reset_index(drop=True)
        )

        return edges

    def _load_predicted_flow(self) -> dict[tuple[str, int], float]:
        """Create predicted flow lookup by site and 15-minute interval."""
        prediction_file = (
            self.root
            / "saved_models"
            / "model_test_predictions.csv"
        )

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Cannot find {prediction_file}. "
                "Run the model evaluation output generation first."
            )

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
            + predictions["target_timestamp"].dt.minute // 15
        )

        # Average the same time interval across available testing days.
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
        """Return all SCATS nodes available in the route graph."""
        return set(self.edges["from_site"]) | set(self.edges["to_site"])

    def neighbours(self, site_id: str) -> list[str]:
        """Return all next nodes directly reachable from one SCATS site."""
        site_id = _normalise_site_id(site_id)

        return (
            self.edges.loc[
                self.edges["from_site"] == site_id,
                "to_site",
            ]
            .tolist()
        )

    def get_segment_detail(
        self,
        from_site: str,
        to_site: str,
        entry_time: datetime,
    ) -> dict:
        """
        Calculate detailed dynamic travel cost for one route segment.

        Flow is taken from the starting SCATS site at the interval containing
        the time the vehicle enters this segment.
        """
        from_site = _normalise_site_id(from_site)
        to_site = _normalise_site_id(to_site)

        edge = self.edges[
            (self.edges["from_site"] == from_site)
            & (self.edges["to_site"] == to_site)
        ]

        if edge.empty:
            raise ValueError(f"No edge found from {from_site} to {to_site}.")

        interval = _time_interval(entry_time)
        flow_key = (from_site, interval)

        if flow_key not in self.flow_lookup:
            raise ValueError(
                f"No predicted flow found for SCATS site {from_site} "
                f"at interval {_interval_label(interval)}."
            )

        predicted_flow_15_min = self.flow_lookup[flow_key]

        # Model flow is per 15 minutes; speed conversion uses hourly flow.
        predicted_flow_per_hour = predicted_flow_15_min * 4

        speed_kmh = flow_to_speed(predicted_flow_per_hour)
        distance_km = float(edge.iloc[0]["distance_km"])

        segment_time = (
            (distance_km / speed_kmh) * 60
            + INTERSECTION_DELAY_MINUTES
        )

        arrival_time = entry_time + timedelta(minutes=segment_time)

        road_name = (
            str(edge.iloc[0]["road_name"])
            if "road_name" in edge.columns
            else ""
        )

        return {
            "from_site": from_site,
            "to_site": to_site,
            "road_name": road_name,
            "entry_time": entry_time,
            "traffic_interval": _interval_label(interval),
            "predicted_flow_15_min": predicted_flow_15_min,
            "predicted_flow_per_hour": predicted_flow_per_hour,
            "speed_kmh": speed_kmh,
            "distance_km": distance_km,
            "intersection_delay_minutes": INTERSECTION_DELAY_MINUTES,
            "segment_time_minutes": segment_time,
            "arrival_time": arrival_time,
        }

    def edge_travel_time(
        self,
        from_site: str,
        to_site: str,
        entry_time: datetime,
    ) -> float:
        """Return only the dynamic travel time used by the search algorithm."""
        detail = self.get_segment_detail(
            from_site=from_site,
            to_site=to_site,
            entry_time=entry_time,
        )

        return detail["segment_time_minutes"]