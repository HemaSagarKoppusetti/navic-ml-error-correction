import pandas as pd
import numpy as np

# -------- FILE PATHS --------
INPUT_FILE = "dataset/processed/final_features.csv"
OUTPUT_FILE = "dataset/processed/ml_dataset.csv"


# -------- GLOBAL REFERENCE (FIXED) --------
REF_LAT = 18.56334100
REF_LON = 73.83323500


# -------- TRUE POSITIONS --------
GT_FLAT = (18.56334100, 73.83323500)
GT_INCLINED = (18.56333783, 73.83323233)


# -------- LAT/LON → XY --------
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371000

    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R

    return x, y


def create_labels():

    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Creating corrected labels...")

    x_raw_list = []
    y_raw_list = []
    x_gt_list = []
    y_gt_list = []
    error_x_list = []
    error_y_list = []

    for _, row in df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        # -------- RAW POSITION --------
        x_raw, y_raw = latlon_to_xy(lat, lon, REF_LAT, REF_LON)

        # -------- SELECT TRUE POSITION --------
        if row["is_inclined"] == 0:
            gt_lat, gt_lon = GT_FLAT
        else:
            gt_lat, gt_lon = GT_INCLINED

        # -------- TRUE POSITION IN SAME FRAME --------
        x_gt, y_gt = latlon_to_xy(gt_lat, gt_lon, REF_LAT, REF_LON)

        # -------- ERROR CALCULATION --------
        error_x = x_raw - x_gt
        error_y = y_raw - y_gt

        x_raw_list.append(x_raw)
        y_raw_list.append(y_raw)
        x_gt_list.append(x_gt)
        y_gt_list.append(y_gt)
        error_x_list.append(error_x)
        error_y_list.append(error_y)

    # -------- ADD TO DATAFRAME --------
    df["x"] = x_raw_list
    df["y"] = y_raw_list
    df["gt_x"] = x_gt_list
    df["gt_y"] = y_gt_list
    df["error_x"] = error_x_list
    df["error_y"] = error_y_list

    # -------- MAGNITUDE --------
    df["original_error"] = np.sqrt(df["error_x"]**2 + df["error_y"]**2)

    print("Saving corrected dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ FIXED Step 3 completed!")
    print(f"Saved as: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    create_labels()