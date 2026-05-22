import math
import pandas as pd


def flow_to_speed(flow):
    # formula from conversion document:
    # flow = -1.46484375 * speed^2 + 93.75 * speed

    A = -1.46484375  # -1500 / 1024, from conversion doc
    B = 93.75
    speed_limit = 60

    if flow <= 351:
        return speed_limit

    discriminant = B**2 - 4 * A * (-flow)

    if discriminant < 0:
        return 32

    speed1 = (-B + math.sqrt(discriminant)) / (2 * A)
    speed2 = (-B - math.sqrt(discriminant)) / (2 * A)

    possible_speeds = [
        speed for speed in [speed1, speed2]
        if speed > 0
    ]

    if not possible_speeds:
        return 32

    speed = max(possible_speeds)

    return min(speed, speed_limit)


# load files
edges = pd.read_csv("data/route_map/final_edges.csv")
predictions = pd.read_csv("saved_models/model_test_predictions.csv")

# average predicted flow per model per site
site_flow = (
    predictions
    .groupby(["Model", "site_id"])["predicted_total_flow"]
    .mean()
    .reset_index()
    .rename(columns={"predicted_total_flow": "predicted_flow"})
)

# convert from per-15-min to per-hour
site_flow["predicted_flow"] = site_flow["predicted_flow"] * 4

# match predicted flow to each edge using FROM site
travel_cost = edges.merge(
    site_flow,
    left_on="from_site",
    right_on="site_id",
    how="left"
)

missing = travel_cost["predicted_flow"].isna().sum()

if missing:
    print("Warning:", missing, "edges have no predicted flow")

# convert flow to speed
travel_cost["speed_kmh"] = travel_cost["predicted_flow"].apply(
    lambda flow: flow_to_speed(flow) if pd.notna(flow) else 60
)

# travel time in minutes
# time = distance / speed
# multiply by 60 to convert hours to minutes
# add 0.5 minute = 30 seconds intersection delay
travel_cost["travel_time"] = (
    travel_cost["distance_km"] / travel_cost["speed_kmh"]
) * 60 + 0.5

travel_cost = travel_cost[
    [
        "Model",
        "from_site",
        "to_site",
        "road_name",
        "distance_km",
        "predicted_flow",
        "speed_kmh",
        "travel_time"
    ]
]

travel_cost.to_csv(
    "data/route_map/travel_cost.csv",
    index=False
)

print("Travel cost rows:", len(travel_cost))
print(travel_cost.head())