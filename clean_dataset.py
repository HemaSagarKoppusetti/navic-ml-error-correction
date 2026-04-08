import pandas as pd

# -------- FILE PATHS --------
INPUT_FILE = "dataset/processed/merged_dataset_noisy.csv"
OUTPUT_FILE = "dataset/processed/cleaned_dataset.csv"


def clean_dataset():

    print("Reading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Initial rows:", len(df))

    # -------- 1. HANDLE MISSING VALUES --------

    # mean_snr → fill with median
    df["mean_snr"] = df["mean_snr"].fillna(df["mean_snr"].median())

    # mean_elevation → fill with median
    df["mean_elevation"] = df["mean_elevation"].fillna(df["mean_elevation"].median())

    # sat_count → fill with 0
    df["sat_count"] = df["sat_count"].fillna(0)

    # -------- 2. ORIENTATION FIX --------

    # forward fill (important for time continuity)
    # forward fill (important for time continuity)
    df["roll"] = df["roll"].ffill()
    df["pitch"] = df["pitch"].ffill()
    df["yaw"] = df["yaw"].ffill()

    # -------- 3. DROP INVALID ROWS --------

    # remove rows with missing position
    df = df.dropna(subset=["latitude", "longitude"])

    # -------- 4. SORT BY TIME --------
    df = df.sort_values(by="timestamp")

    print("Cleaned rows:", len(df))

    # -------- SAVE --------
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Step 2.3 completed!")
    print(f"Saved as: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    clean_dataset()