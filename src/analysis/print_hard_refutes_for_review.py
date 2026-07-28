import pandas as pd

PATH = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis/persistent_hard_refutes_manual_review.csv"
)

df = pd.read_csv(PATH)

for i, row in df.iterrows():
    print("\n" + "=" * 100)
    print(f"REVIEW NUMBER: {i + 1}")
    print(f"ID: {row['id']}")
    print(f"LANGUAGE: {row['language']}")
    print(f"GOLD LABEL: {row['gold_label']}")

    print("\nCLAIM:")
    print(row["claim"])

    print("\nGOLD EVIDENCE:")
    print(row["gold_evidence"])

print("\n" + "=" * 100)
print("TOTAL EXAMPLES:", len(df))