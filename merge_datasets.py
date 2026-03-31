import os
import pandas as pd

# -------- PATHS --------
INPUT_DIR = "dataset/processed/parsed"
OUTPUT_FILE = "dataset/processed/merged.csv"

# -------- MAIN FUNCTION --------
def merge_all_files():
    all_dataframes = []

    for file in os.listdir(INPUT_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(INPUT_DIR, file)

            print(f"Reading: {file}")

            df = pd.read_csv(file_path)

            # Add source file column
            df["source_file"] = file.replace(".csv", "")

            all_dataframes.append(df)

    # Merge all
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Save output
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Merging complete!")
    print(f"Total rows: {len(merged_df)}")
    print(f"Saved at: {OUTPUT_FILE}")


# -------- RUN --------
if __name__ == "__main__":
    merge_all_files()