import re
import math
import pandas as pd

def parse_location(location):
    # get road name, direction and reference road from location
    text = str(location).strip().upper().replace(" ", "_")
    match = re.match(r"(.+?)_(NE|NW|SE|SW|N|S|E|W)_OF_(.+)", text)
    if match:
        return pd.Series([match.group(1), match.group(2), match.group(3)])
    return pd.Series([None, None, None])


def get_distance(lat1, lon1, lat2, lon2):
    # calculate distance in km
    r = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


df = pd.read_csv("data/processed/traffic_direction_timeseries.csv")

df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["flow"]      = pd.to_numeric(df["flow"],      errors="coerce")

df[["road_name", "direction", "reference_road"]] = df["location"].apply(parse_location)

df = df[
    df[["latitude", "longitude", "flow", "road_name", "direction"]].notna().all(axis=1)
].copy()


# create one node for each SCATS site
# use average coordinates
nodes = (
    df[["site_id", "location", "latitude", "longitude"]]
    .groupby("site_id", as_index=False)
    .agg(location=("location", "first"),
         latitude=("latitude", "mean"),
         longitude=("longitude", "mean"))
    .sort_values("site_id")
    .reset_index(drop=True)
)


# edges 
# group sites by road
# sort by coordinate
# connect nearby sites
# save both directions

edges = []

for road_name, group in df.groupby("road_name"):
    group = group.drop_duplicates(subset=["site_id"]).copy()
    if len(group) < 2:
        continue

    # sort sites along whichever axis spans more distance
    lon_range = group["longitude"].max() - group["longitude"].min()
    lat_range = group["latitude"].max()  - group["latitude"].min()
    sort_col  = "longitude" if lon_range >= lat_range else "latitude"
    group = group.sort_values(sort_col).reset_index(drop=True)

    for i in range(len(group) - 1):
        a = group.iloc[i]
        b = group.iloc[i + 1]

        dist      = get_distance(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        mean_flow = (a["flow"] + b["flow"]) / 2   # placeholder; replaced by ML prediction at runtime

        edges.append({
            "from_site": a["site_id"],
            "to_site": b["site_id"],

            "road_name": road_name,
            "distance_km": round(dist, 4),
            "mean_flow": mean_flow,

            "from_location": a["location"],
            "to_location": b["location"],

            "from_direction": a["direction"],
            "to_direction": b["direction"]
        })

        edges.append({
            "from_site": b["site_id"],
            "to_site": a["site_id"],

            "road_name": road_name,
            "distance_km": round(dist, 4),
            "mean_flow": mean_flow,

            "from_location": b["location"],
            "to_location": a["location"],

            "from_direction": b["direction"],
            "to_direction": a["direction"]
        })

edges_df = (
    pd.DataFrame(edges)
    .drop_duplicates(subset=["from_site", "to_site", "road_name"])
    .reset_index(drop=True)
)

connected = set(edges_df["from_site"]) | set(edges_df["to_site"])
isolated  = set(nodes["site_id"]) - connected
if isolated:
    print(f"Warning: {len(isolated)} isolated node(s) — no edges found: {isolated}")


# save file
nodes.to_csv("data/route_map/final_nodes.csv", index=False)
edges_df.to_csv("data/route_map/final_edges.csv", index=False)

print(f"Nodes : {len(nodes)}")
print(f"Edges : {len(edges_df)}")
print("Saved to data/route_map/")