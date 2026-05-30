from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from graph.cus1 import find_routes


# -------------------------------------------------------------------------
# Project paths and configuration
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.json"
NODES_FILE = PROJECT_ROOT / "data" / "route_map" / "final_nodes.csv"
EDGES_FILE = PROJECT_ROOT / "data" / "route_map" / "final_edges.csv"
PROCESSED_TRAFFIC_FILE = (
    PROJECT_ROOT / "data" / "processed" / "traffic_direction_timeseries.csv"
)

AVAILABLE_MODELS = ("GRU", "CNN", "LSTM")
CAPACITY_FLOW_PER_HOUR = 1500.0


def load_config() -> dict:
    """Load GUI default values from config.json."""
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

    if not text:
        raise ValueError("SCATS site ID cannot be empty.")

    return text.zfill(4)


def load_map_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load valid map nodes and connections for GUI visualisation."""
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

    # Visualisation only: remove coordinate outliers outside Melbourne.
    nodes = nodes[
        nodes["longitude"].between(144.0, 146.0)
        & nodes["latitude"].between(-39.0, -37.0)
    ].copy()

    valid_sites = set(nodes["site_id"])

    edges = edges[
        edges["from_site"].isin(valid_sites)
        & edges["to_site"].isin(valid_sites)
    ].copy()

    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def load_site_location_reference() -> dict[str, str]:
    """
    Load SCATS site ID to intersection/location text from processed data.

    The GUI input remains SCATS ID based. This reference table only helps
    the user find the correct SCATS ID for a location/intersection.
    """
    if not PROCESSED_TRAFFIC_FILE.exists():
        return {}

    traffic_df = pd.read_csv(PROCESSED_TRAFFIC_FILE)

    if (
        "site_id" not in traffic_df.columns
        or "location" not in traffic_df.columns
    ):
        return {}

    traffic_df = traffic_df[["site_id", "location"]].dropna().copy()
    traffic_df["site_id"] = traffic_df["site_id"].apply(normalise_site_id)
    traffic_df["location"] = traffic_df["location"].astype(str).str.strip()

    location_reference = (
        traffic_df
        .drop_duplicates()
        .groupby("site_id")["location"]
        .apply(lambda values: " / ".join(sorted(set(values))))
        .to_dict()
    )

    return location_reference


CONFIG = load_config()

DEFAULT_MODEL = str(CONFIG["default_model"]).upper()
DEFAULT_DEPARTURE_TIME = str(CONFIG["default_departure_time"])
DEFAULT_NUMBER_OF_ROUTES = int(CONFIG["default_number_of_routes"])
MAXIMUM_NUMBER_OF_ROUTES = int(CONFIG["maximum_number_of_routes"])

if DEFAULT_MODEL not in AVAILABLE_MODELS:
    raise ValueError(
        f"default_model in config.json must be one of: "
        f"{', '.join(AVAILABLE_MODELS)}."
    )

nodes_df, edges_df = load_map_data()
site_location_reference = load_site_location_reference()

node_coordinates = {
    row.site_id: (row.longitude, row.latitude)
    for row in nodes_df.itertuples(index=False)
}

SCATS_DISPLAY_ROWS = []

for node in nodes_df.sort_values("site_id").itertuples(index=False):
    location_text = site_location_reference.get(
        node.site_id,
        "Location not available",
    )

    SCATS_DISPLAY_ROWS.append(
        f"{node.site_id:<8} | {location_text}"
    )

current_routes = []


# -------------------------------------------------------------------------
# Display formatting
# -------------------------------------------------------------------------

def format_duration(minutes: float) -> str:
    """Convert decimal minutes into MM:SS format."""
    total_seconds = int(round(minutes * 60))
    minute_part = total_seconds // 60
    second_part = total_seconds % 60

    return f"{minute_part:02d}:{second_part:02d}"


def traffic_state(hourly_flow: float) -> str:
    """Return the capacity state used for route line colouring."""
    if hourly_flow > CAPACITY_FLOW_PER_HOUR:
        return "CONGESTED / RED"

    return "UNDER CAPACITY / GREEN"


def route_line_colour(hourly_flow: float) -> str:
    """Return the route line colour for one traffic segment."""
    if hourly_flow > CAPACITY_FLOW_PER_HOUR:
        return "#d62728"

    return "#2ca02c"


# -------------------------------------------------------------------------
# Map visualisation
# -------------------------------------------------------------------------

def draw_network(selected_route=None) -> None:
    """
    Draw the network and highlight the selected route by traffic state.

    Grey lines represent the network.
    Green route segments are under capacity.
    Red route segments are congested / over capacity.
    """
    map_axis.clear()

    origin = None
    destination = None

    if selected_route is not None and selected_route.path:
        origin = selected_route.path[0]
        destination = selected_route.path[-1]

    # final_edges.csv stores both directions.
    # Draw each physical connection once as the grey background network.
    background_edges = edges_df.copy()
    background_edges["pair_key"] = background_edges.apply(
        lambda row: tuple(sorted([row["from_site"], row["to_site"]])),
        axis=1,
    )
    background_edges = background_edges.drop_duplicates("pair_key")

    for edge in background_edges.itertuples(index=False):
        if (
            edge.from_site not in node_coordinates
            or edge.to_site not in node_coordinates
        ):
            continue

        x_start, y_start = node_coordinates[edge.from_site]
        x_end, y_end = node_coordinates[edge.to_site]

        map_axis.plot(
            [x_start, x_end],
            [y_start, y_end],
            color="grey",
            linewidth=1.1,
            alpha=0.45,
            zorder=1,
        )

    # Draw the selected route using green/red capacity colours.
    if selected_route is not None:
        for segment in selected_route.segments:
            from_site = segment["from_site"]
            to_site = segment["to_site"]
            hourly_flow = float(segment["predicted_flow_per_hour"])

            if (
                from_site not in node_coordinates
                or to_site not in node_coordinates
            ):
                continue

            x_start, y_start = node_coordinates[from_site]
            x_end, y_end = node_coordinates[to_site]

            line_colour = route_line_colour(hourly_flow)

            map_axis.annotate(
                "",
                xy=(x_end, y_end),
                xytext=(x_start, y_start),
                arrowprops={
                    "arrowstyle": "->",
                    "color": line_colour,
                    "linewidth": 3.3,
                    "alpha": 1.0,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
                zorder=3,
            )

    # Draw nodes and node labels.
    for node in nodes_df.itertuples(index=False):
        if node.site_id == origin:
            node_colour = "#1f77b4"
            node_size = 78
        elif node.site_id == destination:
            node_colour = "orange"
            node_size = 78
        else:
            node_colour = "#78b7df"
            node_size = 48

        map_axis.scatter(
            node.longitude,
            node.latitude,
            s=node_size,
            color=node_colour,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )

        map_axis.text(
            node.longitude,
            node.latitude,
            node.site_id,
            fontsize=7,
            ha="left",
            va="bottom",
            zorder=5,
        )

    if selected_route is None:
        title = "SCATS Road Network"
    else:
        title = (
            f"Route {selected_route.rank}: {selected_route.route_text}  |  "
            f"{selected_route.total_travel_time:.2f} minutes "
            f"({format_duration(selected_route.total_travel_time)})"
        )

    map_axis.set_title(title, fontsize=11, pad=8)
    map_axis.set_xlabel("Longitude")
    map_axis.set_ylabel("Latitude")
    map_axis.grid(alpha=0.2)

    map_axis.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    map_axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    map_axis.ticklabel_format(style="plain", axis="both")

    legend_items = [
        Line2D(
            [0],
            [0],
            color="grey",
            linewidth=1.3,
            label="Road network",
        ),
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            linewidth=3.3,
            label="Under capacity",
        ),
        Line2D(
            [0],
            [0],
            color="#d62728",
            linewidth=3.3,
            label="Congested / over capacity",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#1f77b4",
            markeredgecolor="white",
            label="Origin",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="orange",
            markeredgecolor="white",
            label="Destination",
        ),
    ]

    map_axis.legend(
        handles=legend_items,
        loc="best",
        fontsize=8,
    )

    map_figure.tight_layout()
    map_canvas.draw()


def update_route_selector(routes) -> None:
    """Update route dropdown and show the fastest returned route."""
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
    """Draw the route currently selected from the returned routes."""
    selected_index = route_selector.current()

    if 0 <= selected_index < len(current_routes):
        draw_network(current_routes[selected_index])


# -------------------------------------------------------------------------
# Route calculation and user controls
# -------------------------------------------------------------------------

def format_gui_results(routes) -> str:
    """Format route summaries and segment traffic states for the GUI."""
    output_lines = []

    for route in routes:
        output_lines.extend(
            [
                f"Route {route.rank}: {route.route_text}",
                (
                    "Estimated Travel Time: "
                    f"{route.total_travel_time:.2f} minutes "
                    f"({format_duration(route.total_travel_time)})"
                ),
            ]
        )

        if route.segments:
            output_lines.append("Segment Traffic States:")

            for segment in route.segments:
                hourly_flow = float(segment["predicted_flow_per_hour"])

                output_lines.append(
                    f"  {segment['from_site']} → {segment['to_site']}: "
                    f"{hourly_flow:.2f} veh/hour "
                    f"[{traffic_state(hourly_flow)}]"
                )

        output_lines.append("")

    return "\n".join(output_lines).rstrip()


def update_model_display(event=None) -> None:
    """Update the subtitle when model selection changes."""
    subtitle_label.configure(
        text=(
            "Dynamic route estimation using predicted traffic flow "
            f"from selected model: {model_selector.get()}"
        )
    )


def update_scats_reference_filter(*args) -> None:
    """Filter the SCATS ID reference list by site ID or location text."""
    search_text = scats_search_var.get().strip().lower()

    site_reference_text.configure(state="normal")
    site_reference_text.delete("1.0", tk.END)

    site_reference_text.insert(
        tk.END,
        "SCATS ID | Intersection / Location\n",
    )
    site_reference_text.insert(
        tk.END,
        "-" * 70 + "\n",
    )

    matched_rows = [
        display_row
        for display_row in SCATS_DISPLAY_ROWS
        if search_text in display_row.lower()
    ]

    for display_row in matched_rows:
        site_reference_text.insert(tk.END, display_row + "\n")

    site_reference_text.insert(
        tk.END,
        f"\nShowing {len(matched_rows)} of {len(SCATS_DISPLAY_ROWS)} sites."
    )

    site_reference_text.configure(state="disabled")


def find_best_routes() -> None:
    """Find routes using the input and the selected prediction model."""
    global current_routes

    origin = origin_entry.get().strip()
    destination = destination_entry.get().strip()
    departure_time = departure_entry.get().strip()
    selected_model = model_selector.get().strip().upper()

    if not origin or not destination or not departure_time:
        result_text.delete("1.0", tk.END)
        result_text.insert(
            tk.END,
            "Please enter origin, destination and departure time.",
        )
        return

    if selected_model not in AVAILABLE_MODELS:
        result_text.delete("1.0", tk.END)
        result_text.insert(
            tk.END,
            f"Please select a model: {', '.join(AVAILABLE_MODELS)}.",
        )
        return

    try:
        current_routes = find_routes(
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            k=int(route_count.get()),
            model_name=selected_model,
        )

        output = (
            f"Origin: {normalise_site_id(origin)}\n"
            f"Destination: {normalise_site_id(destination)}\n"
            f"Departure Time: {departure_time}\n"
            f"Model Used: {selected_model}\n\n"
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
    """Reset inputs and model selection to config.json defaults."""
    global current_routes

    current_routes = []

    origin_entry.delete(0, tk.END)
    destination_entry.delete(0, tk.END)

    departure_entry.delete(0, tk.END)
    departure_entry.insert(0, DEFAULT_DEPARTURE_TIME)

    model_selector.set(DEFAULT_MODEL)
    update_model_display()

    route_count.set(str(DEFAULT_NUMBER_OF_ROUTES))

    result_text.delete("1.0", tk.END)

    route_selector.set("")
    route_selector["values"] = []

    scats_search_var.set("")

    draw_network()


# -------------------------------------------------------------------------
# GUI layout
# -------------------------------------------------------------------------

root = tk.Tk()
root.title(CONFIG["window_title"])
root.geometry(CONFIG["window_size"])
root.minsize(1250, 760)


title_label = ttk.Label(
    root,
    text="Traffic-Based Route Guidance System",
    font=("Arial", 19, "bold"),
)
title_label.pack(pady=(12, 2))

subtitle_label = ttk.Label(
    root,
    text=(
        "Dynamic route estimation using predicted traffic flow "
        f"from selected model: {DEFAULT_MODEL}"
    ),
)
subtitle_label.pack(pady=(0, 10))


main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))

main_frame.columnconfigure(0, weight=0)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)


# -------------------------------------------------------------------------
# Left panel: inputs, parameter settings, reference list and route results
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
origin_entry = ttk.Entry(input_frame, width=25)
origin_entry.grid(row=1, column=0, sticky="w", pady=(0, 8))


ttk.Label(input_frame, text="Destination SCATS Site:").grid(
    row=2,
    column=0,
    sticky="w",
    pady=5,
)
destination_entry = ttk.Entry(input_frame, width=25)
destination_entry.grid(row=3, column=0, sticky="w", pady=(0, 8))


ttk.Label(input_frame, text="Departure Time (HH:MM):").grid(
    row=4,
    column=0,
    sticky="w",
    pady=5,
)
departure_entry = ttk.Entry(input_frame, width=25)
departure_entry.grid(row=5, column=0, sticky="w", pady=(0, 8))
departure_entry.insert(0, DEFAULT_DEPARTURE_TIME)


ttk.Label(input_frame, text="Prediction Model:").grid(
    row=6,
    column=0,
    sticky="w",
    pady=5,
)
model_selector = ttk.Combobox(
    input_frame,
    values=AVAILABLE_MODELS,
    state="readonly",
    width=22,
)
model_selector.grid(row=7, column=0, sticky="w", pady=(0, 8))
model_selector.set(DEFAULT_MODEL)
model_selector.bind("<<ComboboxSelected>>", update_model_display)


ttk.Label(input_frame, text="Number of Routes:").grid(
    row=8,
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
    width=22,
)
route_count.grid(row=9, column=0, sticky="w", pady=(0, 12))
route_count.set(str(DEFAULT_NUMBER_OF_ROUTES))


ttk.Button(
    input_frame,
    text="Find Routes",
    command=find_best_routes,
).grid(row=10, column=0, sticky="ew", pady=(0, 6))

ttk.Button(
    input_frame,
    text="Reset Defaults",
    command=reset_defaults,
).grid(row=11, column=0, sticky="ew")


# -------------------------------------------------------------------------
# SCATS ID reference list
# -------------------------------------------------------------------------

site_reference_frame = ttk.LabelFrame(
    left_panel,
    text="SCATS ID Reference",
    padding=10,
)
site_reference_frame.pack(fill="x", pady=(0, 10))


ttk.Label(
    site_reference_frame,
    text="Search SCATS ID / Location:",
).pack(anchor="w")


scats_search_var = tk.StringVar()

scats_search_entry = ttk.Entry(
    site_reference_frame,
    textvariable=scats_search_var,
    width=56,
)
scats_search_entry.pack(fill="x", pady=(4, 8))


site_reference_body = ttk.Frame(site_reference_frame)
site_reference_body.pack(fill="both", expand=True)


site_reference_text = tk.Text(
    site_reference_body,
    width=56,
    height=8,
    wrap="none",
    font=("Courier New", 10),
)

site_reference_scrollbar = ttk.Scrollbar(
    site_reference_body,
    orient="vertical",
    command=site_reference_text.yview,
)

site_reference_text.configure(
    yscrollcommand=site_reference_scrollbar.set,
)

site_reference_text.pack(side="left", fill="both", expand=True)
site_reference_scrollbar.pack(side="right", fill="y")

scats_search_var.trace_add(
    "write",
    update_scats_reference_filter,
)

update_scats_reference_filter()


# -------------------------------------------------------------------------
# Returned routes and traffic states
# -------------------------------------------------------------------------

results_frame = ttk.LabelFrame(
    left_panel,
    text="Returned Routes and Traffic States",
    padding=10,
)
results_frame.pack(fill="both", expand=True)


result_text = tk.Text(
    results_frame,
    width=56,
    height=16,
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
# Right panel: route visualisation
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
    text="Displayed Route:",
).pack(side="left", padx=(0, 8))


route_selector = ttk.Combobox(
    selector_frame,
    state="readonly",
    width=32,
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


# Draw network when GUI first opens.
draw_network()


root.mainloop()
