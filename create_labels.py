import pandas as pd
import numpy as np

# -------- FILE PATHS --------
INPUT_FILE = "dataset/processed/final_features.csv"
OUTPUT_FILE = "dataset/processed/ml_dataset.csv"


# -------- GROUND TRUTH VALUES --------
GT_FLAT = (18.56334100, 78.83323500)
GT_INCLINED = (18.56333783, 78.83323233)


# -------- LAT/LON → XY CONVERSION --------
def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local XY coordinates (meters)
    """
    R = 6371000

    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R

    return x, y


def create_labels():

    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Creating error labels...")

    x_list = []
    y_list = []
    error_x = []
    error_y = []

    for _, row in df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        # -------- SELECT CORRECT GROUND TRUTH --------
        if row["is_inclined"] == 0:
            gt_lat, gt_lon = GT_FLAT
        else:
            gt_lat, gt_lon = GT_INCLINED

        # -------- CONVERT TO LOCAL XY --------
        x, y = latlon_to_xy(lat, lon, gt_lat, gt_lon)

        # -------- ERROR (OFFSET FROM TRUE POSITION) --------
        err_x = x
        err_y = y

        x_list.append(x)
        y_list.append(y)
        error_x.append(err_x)
        error_y.append(err_y)

    # -------- ADD TO DATAFRAME --------
    df["x"] = x_list
    df["y"] = y_list
    df["error_x"] = error_x
    df["error_y"] = error_y

    print("Saving ML dataset...")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Step 3 COMPLETED SUCCESSFULLY!")
    print(f"Saved as: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    create_labels()