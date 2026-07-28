from pathlib import Path
import pandas as pd

MASTER_PATH = Path("results/analysis/master_results.csv")
OUTPUT_PATH = Path("results/final_experiment_summary.csv")

if not MASTER_PATH.exists():
    raise FileNotFoundError(
        f"Master results file not found: {MASTER_PATH}"
    )

df = pd.read_csv(MASTER_PATH)

required_columns = {
    "Category",
    "Experiment",
    "Accuracy",
    "Macro_F1",
    "Precision",
    "Recall",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Master results file is missing required columns: {sorted(missing)}"
    )

legacy_noncomparable = {
    "XLM-R Claim Only",
    "AfriBERTa Claim Only",
}

bad_rows = df[df["Experiment"].isin(legacy_noncomparable)]

if not bad_rows.empty:
    raise ValueError(
        "Non-comparable legacy encoder rows are still present in "
        f"{MASTER_PATH}:\n{bad_rows.to_string(index=False)}"
    )

summary = df[
    [
        "Category",
        "Experiment",
        "Accuracy",
        "Macro_F1",
        "Precision",
        "Recall",
    ]
].copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_PATH, index=False)

print("\nFINAL EXPERIMENT SUMMARY\n")
print(summary.to_string(index=False))

print("\nSORTED BY ACCURACY\n")
print(
    summary.sort_values(
        "Accuracy",
        ascending=False,
    ).to_string(index=False)
)

print(f"\nSaved to {OUTPUT_PATH}")
