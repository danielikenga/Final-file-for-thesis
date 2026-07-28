import pandas as pd
from sklearn.metrics import classification_report

FILES = {
    "claim_only":
        "results/llm/qwen_claim_only_predictions.csv",

    "gold_evidence":
        "results/llm/qwen_gold_evidence_predictions.csv",

    "gold_evidence_reasoning":
        "results/llm/qwen_gold_evidence_reasoning_predictions.csv",

    "bm25_evidence":
        "results/llm/qwen_bm25_evidence_predictions.csv",

    "adversarial_evidence":
        "results/llm/qwen_adversarial_evidence_predictions.csv",
}

for experiment_name, path in FILES.items():

    print("\n" + "=" * 80)
    print(experiment_name.upper())
    print("=" * 80)

    df = pd.read_csv(path)

    print("\nTOTAL EXAMPLES")
    print(len(df))

    print("\nCLASS DISTRIBUTION (GOLD)")
    print(df["gold_label"].value_counts())

    print("\nCLASS DISTRIBUTION (PREDICTED)")
    print(df["predicted_label"].value_counts())

    print("\nPER-LABEL REPORT")
    print(
        classification_report(
            df["gold_label"],
            df["predicted_label"],
            digits=4
        )
    )

    print("\nLANGUAGE BREAKDOWN")

    for lang in sorted(df["language"].unique()):

        subset = df[df["language"] == lang]

        accuracy = (
            subset["gold_label"]
            ==
            subset["predicted_label"]
        ).mean()

        print(
            f"{lang}: "
            f"n={len(subset)} "
            f"accuracy={accuracy:.4f}"
        )