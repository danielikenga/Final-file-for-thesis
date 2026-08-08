import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUTPUT_DIR = "results/analysis/cross_experiment_errors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["supports", "refutes", "nei"]

EXPERIMENTS = {
    "qwen_1_5b_claim_only": "results/llm/qwen_claim_only_predictions.csv",
    "qwen_1_5b_gold": "results/llm/qwen_gold_evidence_predictions.csv",
    "qwen_1_5b_bm25": "results/llm/qwen_bm25_evidence_predictions.csv",
    "qwen_1_5b_adversarial": "results/llm/qwen_adversarial_evidence_predictions.csv",
    "qwen_14b_gold": "results/llm/qwen14b_gold_evidence_predictions.csv",
    "qwen_14b_translated": "results/llm/qwen14b_translated_gold_evidence_predictions.csv",
}


def normalise_label(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    mapping = {
        "support": "supports",
        "supports": "supports",
        "supported": "supports",
        "refute": "refutes",
        "refutes": "refutes",
        "refuted": "refutes",
        "nei": "nei",
        "not enough information": "nei",
        "not_enough_information": "nei",
    }

    return mapping.get(x, x)


def load_prediction_file(name, path):
    df = pd.read_csv(path)


    if "id" not in df.columns:
        if "query_id" in df.columns:
            df = df.rename(columns={"query_id": "id"})
        else:
            raise ValueError(
                f"{name} has no recognised identifier column. "
                f"Available columns: {df.columns.tolist()}"
            )

    required = {"id", "language", "gold_label", "predicted_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    df = df[["id", "language", "gold_label", "predicted_label"]].copy()

    df["gold_label"] = df["gold_label"].apply(normalise_label)
    df["predicted_label"] = df["predicted_label"].apply(normalise_label)

    df = df.rename(
        columns={
            "predicted_label": f"{name}_prediction"
        }
    )

    return df


merged = None

for name, path in EXPERIMENTS.items():
    print(f"Loading {name}: {path}")

    exp_df = load_prediction_file(name, path)

    if merged is None:
        merged = exp_df
    else:
        merged = merged.merge(
            exp_df[["id", f"{name}_prediction"]],
            on="id",
            how="inner",
        )

print("Merged examples:", len(merged))

merged["language"] = merged["language"].astype(str).str.lower().str.strip()


for name in EXPERIMENTS:
    merged[f"{name}_correct"] = (
        merged[f"{name}_prediction"] == merged["gold_label"]
    )

merged.to_csv(
    os.path.join(OUTPUT_DIR, "merged_cross_experiment_predictions.csv"),
    index=False,
)


overall_rows = []

for name in EXPERIMENTS:
    overall_rows.append({
        "Experiment": name,
        "Accuracy": accuracy_score(
            merged["gold_label"],
            merged[f"{name}_prediction"]
        ),
        "Correct": int(merged[f"{name}_correct"].sum()),
        "Total": len(merged),
    })

overall_df = pd.DataFrame(overall_rows)
overall_df.to_csv(
    os.path.join(OUTPUT_DIR, "overall_accuracy.csv"),
    index=False,
)


language_rows = []

for language, group in merged.groupby("language"):
    for name in EXPERIMENTS:
        language_rows.append({
            "Language": language,
            "Experiment": name,
            "Examples": len(group),
            "Accuracy": accuracy_score(
                group["gold_label"],
                group[f"{name}_prediction"]
            ),
        })

language_df = pd.DataFrame(language_rows)
language_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_language_accuracy.csv"),
    index=False,
)



label_rows = []

for label, group in merged.groupby("gold_label"):
    for name in EXPERIMENTS:
        label_rows.append({
            "Gold_Label": label,
            "Experiment": name,
            "Examples": len(group),
            "Accuracy": accuracy_score(
                group["gold_label"],
                group[f"{name}_prediction"]
            ),
        })

label_df = pd.DataFrame(label_rows)
label_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_label_accuracy.csv"),
    index=False,
)


dist_rows = []

for name in EXPERIMENTS:
    counts = merged[f"{name}_prediction"].value_counts()

    for label in LABELS:
        dist_rows.append({
            "Experiment": name,
            "Predicted_Label": label,
            "Count": int(counts.get(label, 0)),
            "Percentage": counts.get(label, 0) / len(merged),
        })

dist_df = pd.DataFrame(dist_rows)
dist_df.to_csv(
    os.path.join(OUTPUT_DIR, "prediction_distribution.csv"),
    index=False,
)


PAIRS = [
    ("qwen_1_5b_claim_only", "qwen_1_5b_gold"),
    ("qwen_1_5b_gold", "qwen_1_5b_bm25"),
    ("qwen_1_5b_gold", "qwen_1_5b_adversarial"),
    ("qwen_1_5b_gold", "qwen_14b_gold"),
    ("qwen_14b_gold", "qwen_14b_translated"),
]

transition_rows = []

for before, after in PAIRS:
    before_correct = merged[f"{before}_correct"]
    after_correct = merged[f"{after}_correct"]

    conditions = {
        "correct_to_correct": before_correct & after_correct,
        "wrong_to_correct": (~before_correct) & after_correct,
        "correct_to_wrong": before_correct & (~after_correct),
        "wrong_to_wrong": (~before_correct) & (~after_correct),
    }

    for transition, mask in conditions.items():
        transition_rows.append({
            "Before": before,
            "After": after,
            "Transition": transition,
            "Count": int(mask.sum()),
            "Percentage": mask.sum() / len(merged),
        })

transition_df = pd.DataFrame(transition_rows)
transition_df.to_csv(
    os.path.join(OUTPUT_DIR, "pairwise_transition_summary.csv"),
    index=False,
)


correct_cols = [f"{name}_correct" for name in EXPERIMENTS]

merged["num_systems_correct"] = merged[correct_cols].sum(axis=1)

hard_examples = merged[
    merged["num_systems_correct"] == 0
].copy()

easy_examples = merged[
    merged["num_systems_correct"] == len(EXPERIMENTS)
].copy()

hard_examples.to_csv(
    os.path.join(OUTPUT_DIR, "hard_examples_all_systems_wrong.csv"),
    index=False,
)

easy_examples.to_csv(
    os.path.join(OUTPUT_DIR, "easy_examples_all_systems_correct.csv"),
    index=False,
)


for name in EXPERIMENTS:
    report = classification_report(
        merged["gold_label"],
        merged[f"{name}_prediction"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"classification_report_{name}.csv")
    )

    cm = confusion_matrix(
        merged["gold_label"],
        merged[f"{name}_prediction"],
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"gold_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    )

    cm_df.to_csv(
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.csv")
    )


print("\nOVERALL ACCURACY")
print(overall_df.to_string(index=False))

print("\nPAIRWISE TRANSITIONS")
print(transition_df.to_string(index=False))

print("\nPREDICTION DISTRIBUTION")
print(dist_df.to_string(index=False))

print("\nHard examples all systems wrong:", len(hard_examples))
print("Easy examples all systems correct:", len(easy_examples))

print("\nSaved analysis to:")
print(OUTPUT_DIR)