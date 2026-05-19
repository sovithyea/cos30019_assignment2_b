from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raws"
PROCESSED_DIR = BASE_DIR / "processed"

TRAFFIC_FILE = RAW_DIR / "Scats Data October 2006.xls"
SITE_LISTING_FILE = RAW_DIR / "SCATSSiteListingSpreadsheet_VicRoads.xls"
LOCATIONS_FILE = RAW_DIR / "Traffic_Count_Locations_with_LONG_LAT.csv"


# COMMON HELPERS

def ensure_processed_folder():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def check_file_exists(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")


def standardise_site_id(value, width=4):
    if pd.isna(value):
        return None

    value = str(value).strip().replace(".0", "")

    if value.isdigit():
        return value.zfill(width)

    return value


# PART 1 - PROCESS SCATS TRAFFIC FLOW DATA

def process_scats_traffic_data():

    print("\nProcessing 1/3: Scats Data October 2006.xls")
    check_file_exists(TRAFFIC_FILE)

    traffic_raw = pd.read_excel(
        TRAFFIC_FILE,
        sheet_name="Data",
        header=1,
        engine="calamine"
    )

    required_columns = [
        "SCATS Number",
        "Location",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "Date"
    ]

    missing_columns = [col for col in required_columns if col not in traffic_raw.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in traffic data: {missing_columns}")

    interval_columns = [
        col for col in traffic_raw.columns
        if isinstance(col, str) and re.fullmatch(r"V\d{2}", col)
    ]

    if len(interval_columns) != 96:
        raise ValueError(f"Expected 96 interval columns V00-V95, found {len(interval_columns)}")

    traffic_long = traffic_raw.melt(
        id_vars=required_columns,
        value_vars=interval_columns,
        var_name="interval_code",
        value_name="flow"
    )

    traffic_long["site_id"] = traffic_long["SCATS Number"].apply(standardise_site_id)
    traffic_long["date_only"] = pd.to_datetime(traffic_long["Date"]).dt.date
    traffic_long["interval"] = traffic_long["interval_code"].str.extract(r"V(\d{2})").astype(int)

    traffic_long["timestamp"] = (
        pd.to_datetime(traffic_long["date_only"].astype(str))
        + pd.to_timedelta(traffic_long["interval"] * 15, unit="m")
    )

    # Aggregate all detector/direction rows into one total flow per SCATS site per timestamp.
    flow_agg = (
        traffic_long
        .groupby(["site_id", "date_only", "interval", "timestamp"], as_index=False)["flow"]
        .sum()
    )

    # Keep representative location and average coordinates per SCATS site.
    site_info = (
        traffic_long
        .groupby("site_id", as_index=False)
        .agg({
            "Location": "first",
            "NB_LATITUDE": "mean",
            "NB_LONGITUDE": "mean"
        })
    )

    traffic_timeseries = flow_agg.merge(site_info, on="site_id", how="left")

    traffic_timeseries = traffic_timeseries.rename(columns={
        "Location": "location",
        "NB_LATITUDE": "latitude",
        "NB_LONGITUDE": "longitude"
    })

    traffic_timeseries = traffic_timeseries[
        [
            "site_id",
            "location",
            "latitude",
            "longitude",
            "date_only",
            "interval",
            "timestamp",
            "flow"
        ]
    ].sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    traffic_timeseries["is_valid_coordinate"] = (
    traffic_timeseries["longitude"].between(140, 150)
    & traffic_timeseries["latitude"].between(-39.5, -33.5)
    )

    invalid_coords = traffic_timeseries[~traffic_timeseries["is_valid_coordinate"]]

    print("Invalid coordinate records:", len(invalid_coords))
    print("Invalid coordinate sites:", invalid_coords["site_id"].nunique())
    print(
        invalid_coords[
            ["site_id", "location", "latitude", "longitude"]
        ].drop_duplicates()
    )   

    output_path = PROCESSED_DIR / "traffic_timeseries.csv"
    traffic_timeseries.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Shape:", traffic_timeseries.shape)
    print("Sites:", traffic_timeseries["site_id"].nunique())
    print("Dates:", traffic_timeseries["date_only"].nunique())
    print("Missing flow:", traffic_timeseries["flow"].isna().sum())

    return traffic_timeseries


# PART 2 - PROCESS SCATS SITE LISTING DATA

def process_scats_site_listing():

    print("\nProcessing 2/3: SCATSSiteListingSpreadsheet_VicRoads.xls")
    check_file_exists(SITE_LISTING_FILE)

    site_listing = pd.read_excel(
        SITE_LISTING_FILE,
        sheet_name="SCATS Site Numbers",
        header=9,
        engine="calamine"
    )

    site_listing = site_listing.dropna(how="all")

    site_listing.columns = (
        site_listing.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "site_number" not in site_listing.columns:
        raise ValueError("Expected column 'site_number' was not found in site listing data.")

    site_listing["site_id"] = site_listing["site_number"].apply(standardise_site_id)

    useful_columns = [
        "site_id",
        "site_number",
        "location_description",
        "site_type",
        "directory",
        "map_reference"
    ]

    useful_columns = [col for col in useful_columns if col in site_listing.columns]

    site_listing_clean = (
        site_listing[useful_columns]
        .drop_duplicates(subset=["site_id"], keep="first")
        .sort_values("site_id")
        .reset_index(drop=True)
    )

    output_path = PROCESSED_DIR / "scats_site_listing_clean.csv"
    site_listing_clean.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Shape:", site_listing_clean.shape)
    print("Unique site IDs:", site_listing_clean["site_id"].nunique())
    print("Missing location description:", site_listing_clean["location_description"].isna().sum())

    return site_listing_clean


# PART 3 - PROCESS TRAFFIC COUNT LOCATIONS DATA

def process_traffic_count_locations():

    print("\nProcessing 3/3: Traffic_Count_Locations_with_LONG_LAT.csv")
    check_file_exists(LOCATIONS_FILE)

    locations = pd.read_csv(LOCATIONS_FILE)

    locations.columns = (
        locations.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    locations = locations.rename(columns={
        "x": "longitude",
        "y": "latitude",
        "aadt_allve": "aadt_all_vehicles",
        "aadt_truck": "aadt_truck",
        "per_trucks": "percent_trucks"
    })

    if "tfm_id" not in locations.columns:
        raise ValueError("Expected column 'TFM_ID' was not found in traffic count locations data.")

    locations["tfm_id"] = (
        locations["tfm_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    useful_columns = [
        "tfm_id",
        "tfm_desc",
        "site_desc",
        "longitude",
        "latitude",
        "tfm_typ_de",
        "movement_t",
        "declared_r",
        "local_road",
        "aadt_all_vehicles",
        "aadt_truck",
        "percent_trucks",
        "last_year"
    ]

    useful_columns = [col for col in useful_columns if col in locations.columns]

    locations_clean = (
        locations[useful_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    output_path = PROCESSED_DIR / "traffic_count_locations_clean.csv"
    locations_clean.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print("Shape:", locations_clean.shape)
    print("Unique TFM IDs:", locations_clean["tfm_id"].nunique())
    print("Missing longitude:", locations_clean["longitude"].isna().sum())
    print("Missing latitude:", locations_clean["latitude"].isna().sum())

    return locations_clean


# MAIN

def main():
    ensure_processed_folder()

    print("Raw data folder:", RAW_DIR)
    print("Processed data folder:", PROCESSED_DIR)

    process_scats_traffic_data()
    process_scats_site_listing()
    process_traffic_count_locations()

    print("\nAll 3 data files have been processed successfully.")


if __name__ == "__main__":
    main()
