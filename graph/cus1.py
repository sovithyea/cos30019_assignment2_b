from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

from graph.travel_cost import (
    DEFAULT_MODEL,
    TravelTimeEstimator,
    parse_departure_time,
)


MAX_ROUTES = 5


@dataclass(frozen=True)
class RouteResult:
    rank: int
    path: list[str]
    total_travel_time: float
    segments: list[dict]

    @property
    def route_text(self) -> str:
        return " → ".join(self.path)


def _normalise_site_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def run(
    estimator: TravelTimeEstimator,
    origin: str,
    destination: str,
    departure_seconds: float,
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> dict | None:
    # Find one fastest route using dynamic segment travel time
    excluded_nodes = blocked_nodes or set()
    excluded_edges = blocked_edges or set()

    if origin in excluded_nodes:
        return None

    counter = 0
    heap: list[tuple[float, int, str, list[str]]] = [
        (0.0, counter, origin, [origin])
    ]
    best_cost: dict[str, float] = {origin: 0.0}

    while heap:
        total_time, _, current, path = heapq.heappop(heap)

        if total_time > best_cost.get(current, float("inf")):
            continue

        if current == destination:
            return {
                "path": path,
                "total_travel_time": total_time,
            }

        entry_time_seconds = departure_seconds + total_time * 60

        for neighbour in sorted(estimator.neighbours(current)):
            if neighbour in excluded_nodes:
                continue

            if (current, neighbour) in excluded_edges:
                continue

            if neighbour in path:
                continue

            segment_time = estimator.edge_travel_time(
                from_site=current,
                to_site=neighbour,
                entry_time_seconds=entry_time_seconds,
            )

            new_total_time = total_time + segment_time

            if new_total_time < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_total_time
                counter += 1

                heapq.heappush(
                    heap,
                    (
                        new_total_time,
                        counter,
                        neighbour,
                        path + [neighbour],
                    ),
                )

    return None


def _calculate_route_segments(
    path: list[str],
    estimator: TravelTimeEstimator,
    departure_seconds: float,
) -> tuple[float, list[dict]]:
    total_time = 0.0
    segments: list[dict] = []

    for index in range(len(path) - 1):
        entry_time_seconds = departure_seconds + total_time * 60

        segment = estimator.get_segment_detail(
            from_site=path[index],
            to_site=path[index + 1],
            entry_time_seconds=entry_time_seconds,
        )

        segments.append(segment)
        total_time += segment["segment_time_minutes"]

    return total_time, segments


def _find_alternative_routes(
    estimator: TravelTimeEstimator,
    origin: str,
    destination: str,
    departure_seconds: float,
    k: int,
) -> list[list[str]]:
    first_route = run(
        estimator=estimator,
        origin=origin,
        destination=destination,
        departure_seconds=departure_seconds,
    )

    if first_route is None:
        return []

    accepted_paths: list[list[str]] = [first_route["path"]]
    candidates: list[tuple[float, int, list[str]]] = []
    candidate_paths: set[tuple[str, ...]] = set()
    counter = 0

    while len(accepted_paths) < k:
        previous_path = accepted_paths[-1]

        for spur_index in range(len(previous_path) - 1):
            root_path = previous_path[: spur_index + 1]
            spur_node = root_path[-1]

            blocked_edges: set[tuple[str, str]] = set()

            for accepted_path in accepted_paths:
                if (
                    len(accepted_path) > spur_index + 1
                    and accepted_path[: spur_index + 1] == root_path
                ):
                    blocked_edges.add(
                        (
                            accepted_path[spur_index],
                            accepted_path[spur_index + 1],
                        )
                    )

            root_time, _ = _calculate_route_segments(
                path=root_path,
                estimator=estimator,
                departure_seconds=departure_seconds,
            )

            spur_route = run(
                estimator=estimator,
                origin=spur_node,
                destination=destination,
                departure_seconds=departure_seconds + root_time * 60,
                blocked_nodes=set(root_path[:-1]),
                blocked_edges=blocked_edges,
            )

            if spur_route is None:
                continue

            candidate_path = root_path[:-1] + spur_route["path"]
            candidate_key = tuple(candidate_path)

            if candidate_key in candidate_paths:
                continue

            if candidate_path in accepted_paths:
                continue

            candidate_time, _ = _calculate_route_segments(
                path=candidate_path,
                estimator=estimator,
                departure_seconds=departure_seconds,
            )

            counter += 1
            heapq.heappush(
                candidates,
                (candidate_time, counter, candidate_path),
            )
            candidate_paths.add(candidate_key)

        if not candidates:
            break

        _, _, selected_path = heapq.heappop(candidates)
        accepted_paths.append(selected_path)

    return accepted_paths


def find_routes(
    origin: str,
    destination: str,
    departure_time: str,
    k: int = MAX_ROUTES,
    model_name: str = DEFAULT_MODEL,
    root: Path | None = None,
) -> list[RouteResult]:
    # Return up to five routes ranked by predicted travel time
    origin = _normalise_site_id(origin)
    destination = _normalise_site_id(destination)
    departure_seconds = parse_departure_time(departure_time)

    try:
        requested_routes = int(k)
    except (TypeError, ValueError) as error:
        raise ValueError("Number of routes must be an integer.") from error

    requested_routes = max(1, min(requested_routes, MAX_ROUTES))

    estimator = TravelTimeEstimator(
        model_name=model_name,
        root=root,
    )

    valid_nodes = estimator.valid_nodes()

    if origin not in valid_nodes:
        raise ValueError(f"Invalid origin SCATS site: {origin}")

    if destination not in valid_nodes:
        raise ValueError(f"Invalid destination SCATS site: {destination}")

    if origin == destination:
        return [
            RouteResult(
                rank=1,
                path=[origin],
                total_travel_time=0.0,
                segments=[],
            )
        ]

    paths = _find_alternative_routes(
        estimator=estimator,
        origin=origin,
        destination=destination,
        departure_seconds=departure_seconds,
        k=requested_routes,
    )

    if not paths:
        raise ValueError(
            f"No available route from {origin} to {destination} "
            f"for departure time {departure_time}."
        )

    results: list[RouteResult] = []

    for rank, path in enumerate(paths, start=1):
        total_time, segments = _calculate_route_segments(
            path=path,
            estimator=estimator,
            departure_seconds=departure_seconds,
        )

        results.append(
            RouteResult(
                rank=rank,
                path=path,
                total_travel_time=total_time,
                segments=segments,
            )
        )

    return results

def format_duration(minutes: float) -> str:
    """Format decimal minutes as MM:SS."""
    total_seconds = int(round(minutes * 60))
    minute_part = total_seconds // 60
    second_part = total_seconds % 60

    return f"{minute_part:02d}:{second_part:02d}"


def format_routes(
    routes: list[RouteResult],
    show_breakdown: bool = False,
) -> str:
    """Format route output for the GUI or testcase logs."""
    formatted_routes: list[str] = []

    for route in routes:
        lines = [f"Route {route.rank}: {route.route_text}"]

        if show_breakdown and route.segments:
            lines.extend(["", "Segment Breakdown:"])

            for index, segment in enumerate(route.segments, start=1):
                lines.extend(
                    [
                        f"{index}. {segment['from_site']} → {segment['to_site']}",
                        f"   Entry Time:          {segment['entry_time']}",
                        f"   Traffic Interval:    {segment['traffic_interval']}",
                        f"   Predicted Flow:      {segment['predicted_flow_15_min']:.2f} vehicles / 15 min",
                        f"   Hourly Flow:         {segment['predicted_flow_per_hour']:.2f} vehicles / hour",
                        f"   Estimated Speed:     {segment['speed_kmh']:.2f} km/h",
                        f"   Distance:            {segment['distance_km']:.4f} km",
                        f"   Segment Time:        {segment['segment_time_minutes']:.2f} minutes "
                        f"({format_duration(segment['segment_time_minutes'])})",
                        f"   Arrival Time:        {segment['arrival_time']}",
                        "",
                    ]
                )

        lines.append(
            f"Total Estimated Travel Time: {route.total_travel_time:.2f} minutes "
            f"({format_duration(route.total_travel_time)})"
        )

        formatted_routes.append("\n".join(lines))

    return "\n\n".join(formatted_routes)