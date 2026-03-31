import pandas as pd

# -------- FILE PATH --------
INPUT_FILE = "dataset/processed/merged.csv"
OUTPUT_FILE = "dataset/processed/merged_with_angle.csv"


# -------- FUNCTION --------
def extract_angle(source_name):
    """
    Extract angle from filename
    """

    if source_name.startswith("run"):
        return 0  # no inclination

    elif source_name.startswith("angle"):
        try:
            return int(source_name.split("_")[1])
        except:
            return 0

    return 0


def add_angle_feature():

    print("Reading merged dataset...")
    df = pd.read_csv(INPUT_FILE)

    print("Extracting inclination angle...")

    df["inclination_angle"] = df["source_file"].apply(extract_angle)

    print("Saving new dataset...")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Step 2.2 completed!")
    print(f"Saved as: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    add_angle_feature()