from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "LSTM"
MAX_ROUTES = 5

@dataclass(frozen=True)
class RouteResult:
    #Store one returned route and its estimated travel time

    rank: int
    path: list[str]
    total_travel_time: float
    nodes_created: int

    @property
    def route_text(self) -> str:
        return " → ".join(self.path)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalise_site_id(value: object) -> str:
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def load_valid_nodes(root: Path | None = None) -> set[str]:
    #Read all valid SCATS nodes from final_nodes.csv
    base = root if root is not None else _project_root()
    nodes_file = base / "data" / "route_map" / "final_nodes.csv"

    if not nodes_file.exists():
        raise FileNotFoundError(
            f"Cannot find {nodes_file}. Run graph/build_route_map.py first."
        )

    nodes_df = pd.read_csv(nodes_file)

    if "site_id" not in nodes_df.columns:
        raise ValueError("final_nodes.csv must contain a 'site_id' column.")

    return {_normalise_site_id(site_id) for site_id in nodes_df["site_id"]}


def load_travel_time_graph(
    model_name: str = DEFAULT_MODEL,
    root: Path | None = None,
) -> dict[str, list[tuple[str, float]]]:
    #Build a weighted SCATS graph using predicted travel time in minutes
    base = root if root is not None else _project_root()
    cost_file = base / "data" / "route_map" / "travel_cost.csv"

    if not cost_file.exists():
        raise FileNotFoundError(
            f"Cannot find {cost_file}. Run graph/travel_cost.py first."
        )

    costs_df = pd.read_csv(cost_file)

    required = {"Model", "from_site", "to_site", "travel_time"}
    missing = required.difference(costs_df.columns)

    if missing:
        raise ValueError(
            f"travel_cost.csv is missing required columns: {sorted(missing)}"
        )

    #Only the selected model contributes costs to final route planning
    selected = costs_df[
        costs_df["Model"].astype(str).str.upper() == model_name.upper()
    ].copy()

    if selected.empty:
        available = sorted(costs_df["Model"].astype(str).unique())
        raise ValueError(
            f"No travel costs found for model '{model_name}'. Available: {available}"
        )

    selected["from_site"] = selected["from_site"].apply(_normalise_site_id)
    selected["to_site"] = selected["to_site"].apply(_normalise_site_id)
    selected["travel_time"] = pd.to_numeric(
        selected["travel_time"],
        errors="coerce",
    )

    if selected["travel_time"].isna().any():
        raise ValueError("travel_cost.csv contains invalid travel_time values.")

    edges: dict[str, list[tuple[str, float]]] = {}

    for row in selected.itertuples(index=False):
        edges.setdefault(row.from_site, []).append(
            (row.to_site, float(row.travel_time))
        )
        edges.setdefault(row.to_site, [])

    return edges


def run(
    problem: dict,
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
) -> dict:
    #Find one lowest-travel-time route using Custom Search 1
    edges = problem["edges"]
    origin = problem["origin"]
    destinations = problem["destinations"]

    destination_set = set(destinations)
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

    #Priority queue expands the current route with the lowest time first
    heap: list[tuple[float, str, int, list[str]]] = [
        (0.0, origin, counter, [origin])
    ]

    visited: set[str] = set()
    best_cost: dict[str, float] = {origin: 0.0}
    nodes_created = 1

    while heap:
        cost_so_far, current, _, path = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        if current in destination_set:
            return {
                "goal": current,
                "nodes_created": nodes_created,
                "path": path,
                "total_cost": cost_so_far,
            }

        for neighbour, edge_cost in sorted(
            edges.get(current, []),
            key=lambda item: item[0],
        ):
            if neighbour in excluded_nodes:
                continue

            if (current, neighbour) in excluded_edges:
                continue

            #Avoid cycles in one route
            if neighbour in path:
                continue

            #Add the predicted travel time of the next road segment
            new_cost = cost_so_far + edge_cost

            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                counter += 1
                nodes_created += 1

                heapq.heappush(
                    heap,
                    (new_cost, neighbour, counter, path + [neighbour]),
                )

    return {
        "goal": None,
        "nodes_created": nodes_created,
        "path": [],
        "total_cost": float("inf"),
    }


def _path_cost(
    path: list[str],
    costs: dict[tuple[str, str], float],
) -> float:
    #Calculate the total predicted travel time of a complete route
    return sum(
        costs[(path[index], path[index + 1])]
        for index in range(len(path) - 1)
    )


def _find_alternative_routes(
    edges: dict[str, list[tuple[str, float]]],
    origin: str,
    destination: str,
    k: int,
) -> list[dict]:
    #Generate up to k loopless routes
    base_problem = {
        "edges": edges,
        "origin": origin,
        "destinations": [destination],
    }

    first = run(base_problem)

    if first["goal"] is None:
        return []

    accepted = [first]
    candidates: list[tuple[float, int, dict]] = []
    known_candidate_paths: set[tuple[str, ...]] = set()

    edge_costs = {
        (source, target): cost
        for source, neighbours in edges.items()
        for target, cost in neighbours
    }

    counter = 0

    while len(accepted) < k:
        previous_path = accepted[-1]["path"]

        for spur_index in range(len(previous_path) - 1):
            spur_node = previous_path[spur_index]
            root_path = previous_path[: spur_index + 1]

            # Block accepted edges so another valid route can be explored
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

            spur_problem = {
                "edges": edges,
                "origin": spur_node,
                "destinations": [destination],
            }

            spur_result = run(
                spur_problem,
                blocked_nodes=set(root_path[:-1]),
                blocked_edges=blocked_edges,
            )

            if spur_result["goal"] is None:
                continue

            total_path = root_path[:-1] + spur_result["path"]
            path_key = tuple(total_path)

            if path_key in known_candidate_paths:
                continue

            if any(route["path"] == total_path for route in accepted):
                continue

            total_cost = _path_cost(total_path, edge_costs)

            candidate = {
                "goal": destination,
                "nodes_created": spur_result["nodes_created"],
                "path": total_path,
                "total_cost": total_cost,
            }

            counter += 1
            heapq.heappush(
                candidates,
                (total_cost, counter, candidate),
            )
            known_candidate_paths.add(path_key)

        if not candidates:
            break

        _, _, next_route = heapq.heappop(candidates)
        accepted.append(next_route)

    return accepted


def find_routes(
    origin: str,
    destination: str,
    k: int = MAX_ROUTES,
    model_name: str = DEFAULT_MODEL,
    root: Path | None = None,
) -> list[RouteResult]:
    #Return up to five SCATS routes ranked by predicted travel time
    origin = _normalise_site_id(origin)
    destination = _normalise_site_id(destination)

    valid_nodes = load_valid_nodes(root)

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
            )
        ]

    try:
        requested_routes = int(k)
    except (TypeError, ValueError) as error:
        raise ValueError("Number of routes must be an integer.") from error

    #Keep returned routes within the assignment limit
    requested_routes = max(1, min(requested_routes, MAX_ROUTES))

    #Predicted travel time is the weighted graph cost
    edges = load_travel_time_graph(
        model_name=model_name,
        root=root,
    )

    routes = _find_alternative_routes(
        edges=edges,
        origin=origin,
        destination=destination,
        k=requested_routes,
    )

    if not routes:
        raise ValueError(
            f"No available route from {origin} to {destination} "
            f"using {model_name} travel costs."
        )

    return [
        RouteResult(
            rank=index,
            path=route["path"],
            total_travel_time=route["total_cost"],
            nodes_created=route["nodes_created"],
        )
        for index, route in enumerate(routes, start=1)
    ]


def format_routes(routes: list[RouteResult]) -> str:
    return "\n\n".join(
        f"Route {route.rank}: {route.route_text}\n"
        f"Estimated travel time: {route.total_travel_time:.2f} minutes"
        for route in routes
    )


if __name__ == "__main__":
    #Quick manual integration check.
    sample_routes = find_routes("3127", "4063", k=5)
    print(format_routes(sample_routes))