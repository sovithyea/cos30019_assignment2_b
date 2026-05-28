from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from graph.travel_cost import (
    DEFAULT_MODEL,
    TravelTimeEstimator,
    parse_departure_time,
)


MAX_ROUTES = 5


@dataclass(frozen=True)
class RouteResult:
    # Store one final route and its segment-by-segment calculation
    rank: int
    path: list[str]
    total_travel_time: float
    nodes_created: int
    segments: list[dict]

    @property
    def route_text(self) -> str:
        return " → ".join(self.path)


def _normalise_site_id(value: object) -> str:
    """Normalise SCATS IDs such as 970, 970.0 and '0970'."""
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def run(
    problem: dict,
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> dict:
    """
    Find one fastest route using cus1

    Edge cost is calculated using predicted traffic flow at the time 
        the vehicle reaches the start of each segment
    """
    estimator: TravelTimeEstimator = problem["estimator"]
    origin = problem["origin"]
    destinations = set(problem["destinations"])
    departure_time: datetime = problem["departure_time"]

    excluded_nodes = blocked_nodes or set()
    excluded_edges = blocked_edges or set()

    if origin in excluded_nodes:
        return {
            "goal": None,
            "nodes_created": 0,
            "path": [],
            "total_cost": float("inf"),
        }

    counter = 0

    # Lowest accumulated travel time is expanded first.
    heap: list[tuple[float, str, int, list[str]]] = [
        (0.0, origin, counter, [origin])
    ]

    best_cost: dict[str, float] = {origin: 0.0}
    nodes_created = 1

    while heap:
        cost_so_far, current, _, path = heapq.heappop(heap)

        if cost_so_far > best_cost.get(current, float("inf")):
            continue

        if current in destinations:
            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "total_cost": cost_so_far,
            }

        current_time = departure_time + timedelta(minutes=cost_so_far)

        for neighbour in sorted(estimator.neighbours(current)):
            if neighbour in excluded_nodes:
                continue

            if (current, neighbour) in excluded_edges:
                continue

            if neighbour in path:
                continue

            edge_cost = estimator.edge_travel_time(
                from_site=current,
                to_site=neighbour,
                entry_time=current_time,
            )

            new_cost = cost_so_far + edge_cost

            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                counter += 1
                nodes_created += 1

                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        neighbour,
                        counter,
                        path + [neighbour],
                    ),
                )

    return {
        "goal": None,
        "nodes_created": nodes_created,
        "path": [],
        "total_cost": float("inf"),
    }


def _calculate_route_segments(
    path: list[str],
    estimator: TravelTimeEstimator,
    departure_time: datetime,
) -> tuple[float, list[dict]]:
    # Calculate detailed segment output and total time for one final route
    total_time = 0.0
    segments: list[dict] = []

    for index in range(len(path) - 1):
        entry_time = departure_time + timedelta(minutes=total_time)

        detail = estimator.get_segment_detail(
            from_site=path[index],
            to_site=path[index + 1],
            entry_time=entry_time,
        )

        segments.append(detail)
        total_time += detail["segment_time_minutes"]

    return total_time, segments


def _find_alternative_routes(
    estimator: TravelTimeEstimator,
    origin: str,
    destination: str,
    departure_time: datetime,
    k: int,
) -> list[dict]:
    #Generate up to k loopless alternative routes
    first_problem = {
        "estimator": estimator,
        "origin": origin,
        "destinations": [destination],
        "departure_time": departure_time,
    }

    first_route = run(first_problem)

    if first_route["goal"] is None:
        return []

    accepted = [first_route]
    candidates: list[tuple[float, int, dict]] = []
    candidate_paths: set[tuple[str, ...]] = set()
    counter = 0

    while len(accepted) < k:
        previous_path = accepted[-1]["path"]

        for spur_index in range(len(previous_path) - 1):
            spur_node = previous_path[spur_index]
            root_path = previous_path[: spur_index + 1]

            blocked_edges: set[tuple[str, str]] = set()

            for accepted_route in accepted:
                accepted_path = accepted_route["path"]

                if (
                    len(accepted_path) > spur_index
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
                departure_time=departure_time,
            )

            spur_departure = departure_time + timedelta(minutes=root_time)

            spur_problem = {
                "estimator": estimator,
                "origin": spur_node,
                "destinations": [destination],
                "departure_time": spur_departure,
            }

            spur_route = run(
                problem=spur_problem,
                blocked_nodes=set(root_path[:-1]),
                blocked_edges=blocked_edges,
            )

            if spur_route["goal"] is None:
                continue

            total_path = root_path[:-1] + spur_route["path"]
            path_key = tuple(total_path)

            if path_key in candidate_paths:
                continue

            if any(route["path"] == total_path for route in accepted):
                continue

            total_time, _ = _calculate_route_segments(
                path=total_path,
                estimator=estimator,
                departure_time=departure_time,
            )

            candidate = {
                "goal": destination,
                "nodes_created": spur_route["nodes_created"],
                "path": total_path,
                "total_cost": total_time,
            }

            counter += 1
            heapq.heappush(candidates, (total_time, counter, candidate))
            candidate_paths.add(path_key)

        if not candidates:
            break

        _, _, selected_route = heapq.heappop(candidates)
        accepted.append(selected_route)

    return accepted


def find_routes(
    origin: str,
    destination: str,
    departure_time: str,
    k: int = MAX_ROUTES,
    model_name: str = DEFAULT_MODEL,
    root: Path | None = None,
) -> list[RouteResult]:
    """Return up to five dynamic routes ranked by predicted travel time."""
    origin = _normalise_site_id(origin)
    destination = _normalise_site_id(destination)
    departure = parse_departure_time(departure_time)

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
                nodes_created=1,
                segments=[],
            )
        ]

    try:
        requested_routes = int(k)
    except (TypeError, ValueError) as error:
        raise ValueError("Number of routes must be an integer.") from error

    requested_routes = max(1, min(requested_routes, MAX_ROUTES))

    routes = _find_alternative_routes(
        estimator=estimator,
        origin=origin,
        destination=destination,
        departure_time=departure,
        k=requested_routes,
    )

    if not routes:
        raise ValueError(
            f"No available route from {origin} to {destination} "
            f"for departure time {departure_time}."
        )

    results: list[RouteResult] = []

    for index, route in enumerate(routes, start=1):
        total_time, segments = _calculate_route_segments(
            path=route["path"],
            estimator=estimator,
            departure_time=departure,
        )

        results.append(
            RouteResult(
                rank=index,
                path=route["path"],
                total_travel_time=total_time,
                nodes_created=route["nodes_created"],
                segments=segments,
            )
        )

    return results


def format_routes(
    routes: list[RouteResult],
    show_breakdown: bool = False,
) -> str:
    # Format routes for terminal output or GUI display
    route_outputs: list[str] = []

    for route in routes:
        output = [
            f"Route {route.rank}: {route.route_text}",
        ]

        if show_breakdown:
            output.append("\nSegment Breakdown:")

            for index, segment in enumerate(route.segments, start=1):
                output.extend(
                    [
                        f"{index}. {segment['from_site']} → {segment['to_site']}",
                        f"   Entry Time:       {segment['entry_time'].strftime('%H:%M:%S')}",
                        f"   Traffic Interval: {segment['traffic_interval']}",
                        f"   Predicted Flow:   {segment['predicted_flow_15_min']:.2f} vehicles / 15 min",
                        f"   Hourly Flow:      {segment['predicted_flow_per_hour']:.2f} vehicles / hour",
                        f"   Distance:         {segment['distance_km']:.4f} km",
                        f"   Segment Time:     {segment['segment_time_minutes']:.2f} minutes",
                        f"   Arrival Time:     {segment['arrival_time'].strftime('%H:%M:%S')}",
                        "",
                    ]
                )

        output.append(
            f"Total Estimated Travel Time: {route.total_travel_time:.2f} minutes"
        )

        route_outputs.append("\n".join(output))

    return "\n\n".join(route_outputs)


if __name__ == "__main__":
    sample_routes = find_routes(
        origin="3127",
        destination="4063",
        departure_time="08:00",
        k=5,
    )

    print(format_routes(sample_routes, show_breakdown=True))