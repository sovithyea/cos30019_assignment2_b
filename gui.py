import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from graph.cus1 import DEFAULT_MODEL, find_routes, format_routes


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
_FALLBACK_CONFIG = {
    "default_departure_time": "08:00",
    "default_route_count": 5,
}


def _load_config() -> dict:
    """Load config.json, falling back to defaults if the file is missing."""
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        return {**_FALLBACK_CONFIG, **data}
    except (FileNotFoundError, json.JSONDecodeError):
        return _FALLBACK_CONFIG


config = _load_config()


def find_best_routes() -> None:
    """Read user input and display time-dependent route results."""
    origin = origin_entry.get().strip()
    destination = destination_entry.get().strip()
    departure = departure_entry.get().strip()

    if not origin or not destination or not departure:
        result_text.delete("1.0", tk.END)
        result_text.insert(
            tk.END,
            "Please enter origin, destination and departure time.",
        )
        return

    try:
        routes = find_routes(
            origin=origin,
            destination=destination,
            departure_time=departure,
            k=int(route_count.get()),
        )

        output = (
            f"Origin: {origin.zfill(4)}\n"
            f"Destination: {destination.zfill(4)}\n"
            f"Departure Time: {departure}\n"
            f"Model Used: {DEFAULT_MODEL}\n\n"
            f"{format_routes(routes)}"
        )

    except (ValueError, FileNotFoundError) as error:
        output = f"Error: {error}"

    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, output)


def clear_inputs() -> None:
    """Clear all input fields and displayed results."""
    origin_entry.delete(0, tk.END)
    destination_entry.delete(0, tk.END)
    departure_entry.delete(0, tk.END)
    departure_entry.insert(0, config["default_departure_time"])
    route_count.set(str(config["default_route_count"]))
    result_text.delete("1.0", tk.END)


root = tk.Tk()
root.title("Traffic-Based Route Guidance System")
root.geometry("880x680")

title_label = ttk.Label(
    root,
    text="Traffic-Based Route Guidance System",
    font=("Arial", 20, "bold"),
)
title_label.pack(pady=(20, 5))

subtitle_label = ttk.Label(
    root,
    text=(
        f"Time-dependent route estimation using predicted traffic flow "
        f"from {DEFAULT_MODEL}"
    ),
)
subtitle_label.pack(pady=(0, 20))

input_frame = ttk.LabelFrame(
    root,
    text="Route Input",
    padding=15,
)
input_frame.pack(fill="x", padx=30, pady=10)

ttk.Label(input_frame, text="Origin SCATS Site:").grid(
    row=0,
    column=0,
    padx=10,
    pady=8,
    sticky="w",
)
origin_entry = ttk.Entry(input_frame, width=25)
origin_entry.grid(row=0, column=1, padx=10, pady=8)

ttk.Label(input_frame, text="Destination SCATS Site:").grid(
    row=1,
    column=0,
    padx=10,
    pady=8,
    sticky="w",
)
destination_entry = ttk.Entry(input_frame, width=25)
destination_entry.grid(row=1, column=1, padx=10, pady=8)

ttk.Label(input_frame, text="Departure Time (HH:MM):").grid(
    row=2,
    column=0,
    padx=10,
    pady=8,
    sticky="w",
)
departure_entry = ttk.Entry(input_frame, width=25)
departure_entry.grid(row=2, column=1, padx=10, pady=8)
departure_entry.insert(0, config["default_departure_time"])

ttk.Label(input_frame, text="Number of Routes:").grid(
    row=3,
    column=0,
    padx=10,
    pady=8,
    sticky="w",
)
route_count = ttk.Combobox(
    input_frame,
    values=["1", "2", "3", "4", "5"],
    state="readonly",
    width=22,
)
route_count.set(str(config["default_route_count"]))
route_count.grid(row=3, column=1, padx=10, pady=8)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

ttk.Button(
    button_frame,
    text="Find Route",
    command=find_best_routes,
).pack(side="left", padx=8)

ttk.Button(
    button_frame,
    text="Clear",
    command=clear_inputs,
).pack(side="left", padx=8)

result_frame = ttk.LabelFrame(
    root,
    text="Route Results",
    padding=10,
)
result_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=(10, 20),
)

result_text = tk.Text(
    result_frame,
    height=18,
    wrap="word",
)
result_text.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(
    result_frame,
    orient="vertical",
    command=result_text.yview,
)
scrollbar.pack(side="right", fill="y")
result_text.configure(yscrollcommand=scrollbar.set)

root.mainloop()