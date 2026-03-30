import os
import pandas as pd
from collections import defaultdict
from math import isnan

# -------- CONFIG --------
BASE_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed/parsed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- HELPER --------
def safe_float(x):
    try:
        return float(x)
    except:
        return None

# -------- MAIN PARSER --------
def parse_gnss_file(file_path, is_inclined):
    data = defaultdict(lambda: {
        "lat": None,
        "lon": None,
        "alt": None,
        "snr_list": [],
        "elev_list": [],
        "roll": None,
        "pitch": None,
        "yaw": None
    })

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 2:
                continue

            record_type = parts[0]

            # ---------------- FIX ----------------
            if record_type == "Fix":
                try:
                    lat = safe_float(parts[2])
                    lon = safe_float(parts[3])
                    alt = safe_float(parts[4])
                    timestamp = parts[9]

                    if lat is not None and lon is not None:
                        data[timestamp]["lat"] = lat
                        data[timestamp]["lon"] = lon
                        data[timestamp]["alt"] = alt
                except:
                    pass

            # ---------------- STATUS ----------------
            elif record_type == "Status":
                try:
                    timestamp = parts[1]
                    snr = safe_float(parts[6])
                    elevation = safe_float(parts[8])

                    if snr is not None:
                        data[timestamp]["snr_list"].append(snr)

                    if elevation is not None:
                        data[timestamp]["elev_list"].append(elevation)
                except:
                    pass

            # ---------------- ORIENTATION ----------------
            elif record_type == "OrientationDeg":
                try:
                    timestamp = parts[1]
                    yaw = safe_float(parts[3])
                    roll = safe_float(parts[4])
                    pitch = safe_float(parts[5])

                    data[timestamp]["yaw"] = yaw
                    data[timestamp]["roll"] = roll
                    data[timestamp]["pitch"] = pitch
                except:
                    pass

    # -------- AGGREGATE --------
    rows = []

    for ts, d in data.items():
        if d["lat"] is None or len(d["snr_list"]) == 0:
            continue

        mean_snr = sum(d["snr_list"]) / len(d["snr_list"])
        sat_count = len(d["snr_list"])
        mean_elev = sum(d["elev_list"]) / len(d["elev_list"]) if d["elev_list"] else 0

        row = {
            "timestamp": ts,
            "latitude": d["lat"],
            "longitude": d["lon"],
            "altitude": d["alt"],
            "mean_snr": mean_snr,
            "sat_count": sat_count,
            "mean_elevation": mean_elev,
            "roll": d["roll"],
            "pitch": d["pitch"],
            "yaw": d["yaw"],
            "is_inclined": is_inclined
        }

        rows.append(row)

    return pd.DataFrame(rows)


# -------- PROCESS ALL FILES --------
def process_all():
    folders = {
        "data_with_no_inclination": 0,
        "data_with_inclination": 1
    }

    for folder, label in folders.items():
        folder_path = os.path.join(BASE_DIR, folder)

        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)

                print(f"Processing: {file}")

                df = parse_gnss_file(file_path, label)

                output_file = file.replace(".txt", ".csv")
                df.to_csv(os.path.join(OUTPUT_DIR, output_file), index=False)


# -------- RUN --------
if __name__ == "__main__":
    process_all()