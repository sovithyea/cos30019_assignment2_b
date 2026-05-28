from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from graph.cus1 import find_routes


# -------------------------------------------------------------------------
# Project paths and configuration
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.json"
NODES_FILE = PROJECT_ROOT / "data" / "route_map" / "final_nodes.csv"
EDGES_FILE = PROJECT_ROOT / "data" / "route_map" / "final_edges.csv"


def load_config() -> dict:
    """Load GUI defaults from config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CONFIG_FILE}. Please create config.json first."
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "default_model",
        "default_departure_time",
        "default_number_of_routes",
        "maximum_number_of_routes",
        "window_title",
        "window_size",
    }

    missing = required.difference(config)

    if missing:
        raise ValueError(
            f"config.json is missing settings: {sorted(missing)}"
        )

    return config


def normalise_site_id(value: object) -> str:
    """Normalise SCATS IDs into four-digit strings."""
    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text.zfill(4)


def load_map_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SCATS nodes and road connections for map visualisation."""
    if not NODES_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {NODES_FILE}. Run graph/build_route_map.py first."
        )

    if not EDGES_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {EDGES_FILE}. Run graph/build_route_map.py first."
        )

    nodes = pd.read_csv(NODES_FILE)
    edges = pd.read_csv(EDGES_FILE)

    required_nodes = {"site_id", "longitude", "latitude"}
    required_edges = {"from_site", "to_site"}

    missing_nodes = required_nodes.difference(nodes.columns)
    missing_edges = required_edges.difference(edges.columns)

    if missing_nodes:
        raise ValueError(
            f"final_nodes.csv is missing columns: {sorted(missing_nodes)}"
        )

    if missing_edges:
        raise ValueError(
            f"final_edges.csv is missing columns: {sorted(missing_edges)}"
        )

    nodes["site_id"] = nodes["site_id"].apply(normalise_site_id)
    edges["from_site"] = edges["from_site"].apply(normalise_site_id)
    edges["to_site"] = edges["to_site"].apply(normalise_site_id)

    nodes["longitude"] = pd.to_numeric(nodes["longitude"], errors="coerce")
    nodes["latitude"] = pd.to_numeric(nodes["latitude"], errors="coerce")

    if nodes[["longitude", "latitude"]].isna().any().any():
        raise ValueError("final_nodes.csv contains invalid coordinates.")

    return nodes, edges


CONFIG = load_config()

DEFAULT_MODEL = CONFIG["default_model"]
DEFAULT_DEPARTURE_TIME = CONFIG["default_departure_time"]
DEFAULT_NUMBER_OF_ROUTES = int(CONFIG["default_number_of_routes"])
MAXIMUM_NUMBER_OF_ROUTES = int(CONFIG["maximum_number_of_routes"])

nodes_df, edges_df = load_map_data()

# Lookup coordinates quickly when drawing edges and routes.
node_coordinates = {
    row.site_id: (row.longitude, row.latitude)
    for row in nodes_df.itertuples(index=False)
}

current_routes = []


# -------------------------------------------------------------------------
# Map visualisation
# -------------------------------------------------------------------------

def draw_network(selected_route=None) -> None:
    """
    Draw the complete SCATS network.

    All normal connections are grey.
    The currently selected route is highlighted in red.
    """
    map_axis.clear()

    selected_edges: set[tuple[str, str]] = set()
    origin = None
    destination = None

    if selected_route is not None:
        path = selected_route.path

        selected_edges = {
            (path[index], path[index + 1])
            for index in range(len(path) - 1)
        }

        origin = path[0]
        destination = path[-1]

    # Draw all road connections first.
    for edge in edges_df.itertuples(index=False):
        from_site = edge.from_site
        to_site = edge.to_site

        if from_site not in node_coordinates or to_site not in node_coordinates:
            continue

        x_start, y_start = node_coordinates[from_site]
        x_end, y_end = node_coordinates[to_site]

        is_selected = (from_site, to_site) in selected_edges

        map_axis.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops={
                "arrowstyle": "->",
                "color": "red" if is_selected else "grey",
                "linewidth": 2.7 if is_selected else 1.0,
                "alpha": 0.95 if is_selected else 0.48,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2 if is_selected else 1,
        )

    # Draw nodes and labels.
    for node in nodes_df.itertuples(index=False):
        site_id = node.site_id

        if site_id == origin:
            node_colour = "green"
            node_size = 68
        elif site_id == destination:
            node_colour = "orange"
            node_size = 68
        else:
            node_colour = "#78b7df"
            node_size = 42

        map_axis.scatter(
            node.longitude,
            node.latitude,
            s=node_size,
            color=node_colour,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

        map_axis.text(
            node.longitude,
            node.latitude,
            site_id,
            fontsize=7,
            ha="left",
            va="bottom",
            zorder=4,
        )

    if selected_route is None:
        title = "SCATS Road Network"
    else:
        title = (
            f"Route {selected_route.rank}: {selected_route.route_text}  |  "
            f"{selected_route.total_travel_time:.2f} minutes"
        )

    map_axis.set_title(title, fontsize=11, pad=8)
    map_axis.set_xlabel("Longitude")
    map_axis.set_ylabel("Latitude")
    map_axis.grid(alpha=0.2)

    map_figure.tight_layout()
    map_canvas.draw()


def update_route_selector(routes) -> None:
    """Update the route dropdown and visualise the fastest returned route."""
    route_selector["values"] = [
        f"Route {route.rank} - {route.total_travel_time:.2f} min"
        for route in routes
    ]

    if routes:
        route_selector.current(0)
        draw_network(routes[0])
    else:
        route_selector.set("")
        draw_network()


def visualise_selected_route(event=None) -> None:
    """Highlight the route currently selected by the user."""
    selected_index = route_selector.current()

    if 0 <= selected_index < len(current_routes):
        draw_network(current_routes[selected_index])


# -------------------------------------------------------------------------
# Route calculation and user controls
# -------------------------------------------------------------------------

def format_gui_results(routes) -> str:
    """Format returned routes for the result text box."""
    output_lines = []

    for route in routes:
        output_lines.append(
            f"Route {route.rank}: {route.route_text}\n"
            f"Estimated Travel Time: {route.total_travel_time:.2f} minutes"
        )

    return "\n\n".join(output_lines)


def find_best_routes() -> None:
    """Find time-dependent routes from the user's input."""
    print(
    "DEBUG INPUT:",
    repr(origin_entry.get()),
    repr(destination_entry.get()),
    repr(departure_entry.get()),
    )
    
    global current_routes

    origin = origin_entry.get().strip()
    destination = destination_entry.get().strip()
    departure_time = departure_entry.get().strip()

    if not origin or not destination or not departure_time:
        result_text.delete("1.0", tk.END)
        result_text.insert(
            tk.END,
            "Please enter origin, destination and departure time.",
        )
        return

    try:
        current_routes = find_routes(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            k=int(route_count.get()),
            model_name=DEFAULT_MODEL,
        )

        output = (
            f"Origin: {normalise_site_id(origin)}\n"
            f"Destination: {normalise_site_id(destination)}\n"
            f"Departure Time: {departure_time}\n"
            f"Model Used: {DEFAULT_MODEL}\n\n"
            f"{format_gui_results(current_routes)}"
        )

        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, output)

        update_route_selector(current_routes)

    except (ValueError, FileNotFoundError) as error:
        current_routes = []

        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Error: {error}")

        route_selector.set("")
        route_selector["values"] = []

        draw_network()


def reset_defaults() -> None:
    """Reset configurable inputs to values from config.json."""
    global current_routes

    current_routes = []

    origin_entry.delete(0, tk.END)
    destination_entry.delete(0, tk.END)

    departure_entry.delete(0, tk.END)
    departure_entry.insert(0, DEFAULT_DEPARTURE_TIME)

    route_count.set(str(DEFAULT_NUMBER_OF_ROUTES))

    result_text.delete("1.0", tk.END)

    route_selector.set("")
    route_selector["values"] = []

    draw_network()


# -------------------------------------------------------------------------
# GUI layout
# -------------------------------------------------------------------------

root = tk.Tk()
root.title(CONFIG["window_title"])
root.geometry(CONFIG["window_size"])
root.minsize(1050, 700)


title_label = ttk.Label(
    root,
    text="Traffic-Based Route Guidance System",
    font=("Arial", 19, "bold"),
)
title_label.pack(pady=(12, 2))

subtitle_label = ttk.Label(
    root,
    text=(
        f"Dynamic route estimation using predicted traffic flow "
        f"from configured model: {DEFAULT_MODEL}"
    ),
)
subtitle_label.pack(pady=(0, 10))


main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))

main_frame.columnconfigure(0, weight=0)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)


# -------------------------------------------------------------------------
# Left panel: input, parameter settings and text results
# -------------------------------------------------------------------------

left_panel = ttk.Frame(main_frame)
left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))


input_frame = ttk.LabelFrame(
    left_panel,
    text="User Input and Parameter Settings",
    padding=12,
)
input_frame.pack(fill="x", pady=(0, 10))


ttk.Label(input_frame, text="Origin SCATS Site:").grid(
    row=0,
    column=0,
    sticky="w",
    pady=5,
)
origin_entry = ttk.Entry(input_frame, width=22)
origin_entry.grid(row=1, column=0, sticky="w", pady=(0, 8))


ttk.Label(input_frame, text="Destination SCATS Site:").grid(
    row=2,
    column=0,
    sticky="w",
    pady=5,
)
destination_entry = ttk.Entry(input_frame, width=22)
destination_entry.grid(row=3, column=0, sticky="w", pady=(0, 8))


ttk.Label(input_frame, text="Departure Time (HH:MM):").grid(
    row=4,
    column=0,
    sticky="w",
    pady=5,
)
departure_entry = ttk.Entry(input_frame, width=22)
departure_entry.grid(row=5, column=0, sticky="w", pady=(0, 8))
departure_entry.insert(0, DEFAULT_DEPARTURE_TIME)


ttk.Label(input_frame, text="Number of Routes:").grid(
    row=6,
    column=0,
    sticky="w",
    pady=5,
)
route_count = ttk.Combobox(
    input_frame,
    values=[
        str(number)
        for number in range(1, MAXIMUM_NUMBER_OF_ROUTES + 1)
    ],
    state="readonly",
    width=19,
)
route_count.grid(row=7, column=0, sticky="w", pady=(0, 12))
route_count.set(str(DEFAULT_NUMBER_OF_ROUTES))


ttk.Button(
    input_frame,
    text="Find Routes",
    command=find_best_routes,
).grid(row=8, column=0, sticky="ew", pady=(0, 6))

ttk.Button(
    input_frame,
    text="Reset Defaults",
    command=reset_defaults,
).grid(row=9, column=0, sticky="ew")


results_frame = ttk.LabelFrame(
    left_panel,
    text="Returned Routes",
    padding=10,
)
results_frame.pack(fill="both", expand=True)


result_text = tk.Text(
    results_frame,
    width=42,
    height=18,
    wrap="word",
)
result_text.pack(side="left", fill="both", expand=True)


result_scrollbar = ttk.Scrollbar(
    results_frame,
    orient="vertical",
    command=result_text.yview,
)
result_scrollbar.pack(side="right", fill="y")

result_text.configure(yscrollcommand=result_scrollbar.set)


# -------------------------------------------------------------------------
# Right panel: geographical route visualisation
# -------------------------------------------------------------------------

visualisation_frame = ttk.LabelFrame(
    main_frame,
    text="SCATS Network Route Visualisation",
    padding=10,
)
visualisation_frame.grid(row=0, column=1, sticky="nsew")


selector_frame = ttk.Frame(visualisation_frame)
selector_frame.pack(fill="x", pady=(0, 6))


ttk.Label(
    selector_frame,
    text="Highlighted Route:",
).pack(side="left", padx=(0, 8))


route_selector = ttk.Combobox(
    selector_frame,
    state="readonly",
    width=30,
)
route_selector.pack(side="left")
route_selector.bind("<<ComboboxSelected>>", visualise_selected_route)


map_figure = Figure(figsize=(8.5, 6.5), dpi=100)
map_axis = map_figure.add_subplot(111)

map_canvas = FigureCanvasTkAgg(
    map_figure,
    master=visualisation_frame,
)
map_canvas.get_tk_widget().pack(fill="both", expand=True)


# Draw the full SCATS network when the GUI first opens.
draw_network()


root.mainloop()