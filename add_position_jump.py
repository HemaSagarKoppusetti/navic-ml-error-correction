import pandas as pd
import numpy as np

# -------- FILE PATHS --------
INPUT_FILE = "dataset/processed/cleaned_dataset.csv"
OUTPUT_FILE = "dataset/processed/final_features.csv"


# -------- HAVERSINE FUNCTION --------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two lat/lon points (in meters)
    """
    R = 6371000  # Earth radius in meters

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def add_position_jump():

    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Calculating position jump...")

    # shift previous values
    df["prev_lat"] = df["latitude"].shift(1)
    df["prev_lon"] = df["longitude"].shift(1)

    # calculate distance
    df["position_jump"] = haversine(
        df["prev_lat"], df["prev_lon"],
        df["latitude"], df["longitude"]
    )

    # first row → no previous
    df["position_jump"] = df["position_jump"].fillna(0)

    # drop helper columns
    df = df.drop(columns=["prev_lat", "prev_lon"])

    print("Saving dataset...")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Step 2.4 completed!")
    print(f"Saved as: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    add_position_jump()