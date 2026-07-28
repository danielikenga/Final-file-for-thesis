import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = "results/analysis/master_results.csv"
OUTPUT_DIR = "results/analysis/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)


# Accuracy Comparison


plt.figure(figsize=(12,6))

plt.bar(df["Experiment"], df["Accuracy"])

plt.xticks(rotation=60, ha="right")

plt.ylabel("Accuracy")

plt.title("Accuracy Across All Experiments")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy_comparison.png"
    ),
    dpi=300
)

plt.close()


# Macro F1 Comparison


plt.figure(figsize=(12,6))

plt.bar(df["Experiment"], df["Macro_F1"])

plt.xticks(rotation=60, ha="right")

plt.ylabel("Macro F1")

plt.title("Macro F1 Across All Experiments")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "macrof1_comparison.png"
    ),
    dpi=300
)

plt.close()

print("Saved figures to", OUTPUT_DIR)