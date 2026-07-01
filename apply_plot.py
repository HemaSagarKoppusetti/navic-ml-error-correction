import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("corrected_dataset.csv")

plt.figure()

plt.plot(df["x"], df["y"], label="Raw Position")
plt.plot(df["corrected_x"], df["corrected_y"], label="Corrected Position")

plt.legend()
plt.title("GNSS Position Correction")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

plt.show()