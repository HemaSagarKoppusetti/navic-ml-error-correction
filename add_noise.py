import pandas as pd
import numpy as np

INPUT_FILE = "dataset/processed/merged_with_angle.csv"
OUTPUT_FILE = "dataset/processed/merged_dataset_noisy.csv"


def add_realistic_noise():

    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Adding structured GNSS noise...")

    # -------- NORMALIZE FEATURES --------
    snr_factor = (30 - df["mean_snr"]) / 30
    elev_factor = (90 - df["mean_elevation"]) / 90

    # handle missing values safely
    snr_factor = snr_factor.fillna(0)
    elev_factor = elev_factor.fillna(0)

    # -------- COMBINE FACTORS --------
    error_scale = (snr_factor + elev_factor) / 2

    # -------- GENERATE NOISE --------
    noise_lat = np.random.normal(0, 1e-5 * error_scale)
    noise_lon = np.random.normal(0, 1e-5 * error_scale)

    # -------- APPLY NOISE --------
    df["latitude"] = df["latitude"] + noise_lat
    df["longitude"] = df["longitude"] + noise_lon

    print("Saving noisy dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Noise added successfully!")
    print(f"Saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    add_realistic_noise()