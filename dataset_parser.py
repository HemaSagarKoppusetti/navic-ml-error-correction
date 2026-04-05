import os
import pandas as pd
from collections import defaultdict

# -------- PATHS --------
BASE_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed/parsed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------- SAFE HELPERS --------
def safe_float(x):
    try:
        return float(x)
    except:
        return None


def normalize_time(ts):
    try:
        return int(int(ts) / 1000)  # convert ms → seconds
    except:
        return None


# -------- MAIN PARSER --------
def parse_gnss_file(file_path, is_inclined):

    # Store data per timestamp
    data = defaultdict(lambda: {
        "lat": None,
        "lon": None,
        "alt": None,
        "snr_list": [],
        "elev_list": [],
        "azimuth_list": [],
        "roll": None,
        "pitch": None,
        "yaw": None
    })

    with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 2:
                continue

            record_type = parts[0]

            # ==========================
            # 📍 GNSS FIX (POSITION)
            # ==========================
            if record_type == "Fix":
                try:
                    lat = safe_float(parts[2])
                    lon = safe_float(parts[3])
                    alt = safe_float(parts[4])
                    ts = normalize_time(parts[8])  # UnixTimeMillis

                    if ts:
                        data[ts]["lat"] = lat
                        data[ts]["lon"] = lon
                        data[ts]["alt"] = alt
                except:
                    continue

            # ==========================
            # 📡 SATELLITE STATUS
            # ==========================
            elif record_type == "Status":
                try:
                    ts = normalize_time(parts[1])

                    snr = safe_float(parts[7])       # Cn0DbHz
                    elev = safe_float(parts[9])      # Elevation
                    azimuth = safe_float(parts[8])   # Azimuth

                    if ts:
                        if snr is not None:
                            data[ts]["snr_list"].append(snr)
                        if elev is not None:
                            data[ts]["elev_list"].append(elev)
                        if azimuth is not None:
                            data[ts]["azimuth_list"].append(azimuth)
                except:
                    continue

            # ==========================
            # 📱 DEVICE ORIENTATION
            # ==========================
            elif record_type == "OrientationDeg":
                try:
                    ts = normalize_time(parts[1])

                    yaw = safe_float(parts[3])
                    roll = safe_float(parts[4])
                    pitch = safe_float(parts[5])

                    if ts:
                        data[ts]["yaw"] = yaw
                        data[ts]["roll"] = roll
                        data[ts]["pitch"] = pitch
                except:
                    continue

    # -------- BUILD FINAL DATA --------
    rows = []

    for ts, d in data.items():

        # Skip if no GPS position
        if d["lat"] is None:
            continue

        # -------- AGGREGATE FEATURES --------
        mean_snr = (
            sum(d["snr_list"]) / len(d["snr_list"])
            if d["snr_list"] else None
        )

        sat_count = len(d["snr_list"])

        mean_elevation = (
            sum(d["elev_list"]) / len(d["elev_list"])
            if d["elev_list"] else None
        )

        mean_azimuth = (
            sum(d["azimuth_list"]) / len(d["azimuth_list"])
            if d["azimuth_list"] else None
        )

        rows.append({
            "timestamp": ts,
            "latitude": d["lat"],
            "longitude": d["lon"],
            "altitude": d["alt"],
            "mean_snr": mean_snr,
            "sat_count": sat_count,
            "mean_elevation": mean_elevation,
            "mean_azimuth": mean_azimuth,
            "roll": d["roll"],
            "pitch": d["pitch"],
            "yaw": d["yaw"],
            "is_inclined": is_inclined
        })

    df = pd.DataFrame(rows)

    print(f"✅ Extracted {len(df)} rows from {os.path.basename(file_path)}")

    return df


# -------- PROCESS ALL FILES --------
def process_all():

    folders = {
        "data_with_no_inclination": 0,
        "data_with_inclination": 1
    }

    for folder, label in folders.items():

        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        print(f"\n📂 Processing folder: {folder}")

        for file in os.listdir(folder_path):

            if file.endswith(".txt"):

                file_path = os.path.join(folder_path, file)

                print(f"\nProcessing: {file}")

                df = parse_gnss_file(file_path, label)

                if df.empty:
                    print("⚠️ No valid GNSS data found")
                    continue

                output_file = file.replace(".txt", ".csv")
                save_path = os.path.join(OUTPUT_DIR, output_file)

                df.to_csv(save_path, index=False)

                print(f"💾 Saved: {save_path}")


# -------- RUN --------
if __name__ == "__main__":
    process_all()