import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
# Paths
ORIGINAL_PATH = (
    "results/llm/qwen14b_gold_evidence_predictions.csv"
)

TRANSLATED_PATH = (
    "results/llm/qwen14b_translated_gold_evidence_predictions.csv"
)

OUTPUT_DIR = "results/analysis/translation_effect"

LABELS = ["supports", "refutes", "nei"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper functions

def normalise_label(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    mapping = {
        "support": "supports",
        "supports": "supports",
        "supported": "supports",

        "refute": "refutes",
        "refutes": "refutes",
        "refuted": "refutes",

        "nei": "nei",
        "not_enough_information": "nei",
        "not enough information": "nei",
    }

    return mapping.get(value, value)


def safe_accuracy(group, prediction_column):

    if len(group) == 0:
        return None

    return accuracy_score(
        group["gold_label"],
        group[prediction_column],
    )
# Load predictions

print("Loading prediction files...")

original = pd.read_csv(ORIGINAL_PATH)
translated = pd.read_csv(TRANSLATED_PATH)

print("Original rows:", len(original))
print("Translated rows:", len(translated))

# Validate required columns

required_columns = {
    "id",
    "language",
    "gold_label",
    "predicted_label",
}

for name, df in [
    ("original", original),
    ("translated", translated),
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} file is missing columns: {missing}"
        )


# Normalise labels

for df in [original, translated]:
    df["gold_label"] = df["gold_label"].apply(normalise_label)
    df["predicted_label"] = df["predicted_label"].apply(
        normalise_label
    )

# Check duplicate IDs

if original["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in original predictions."
    )

if translated["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in translated predictions."
    )


# Merge experiments by example ID

merged = original.merge(
    translated,
    on="id",
    how="inner",
    suffixes=("_original", "_translated"),
)

print("Matched examples:", len(merged))

if len(merged) != len(original) or len(merged) != len(translated):
    print(
        "WARNING: Not all examples matched across both files."
    )


# Validate gold labels and languages


gold_mismatch = (
    merged["gold_label_original"]
    != merged["gold_label_translated"]
)

if gold_mismatch.any():
    raise ValueError(
        f"Gold-label mismatch for "
        f"{gold_mismatch.sum()} examples."
    )

language_mismatch = (
    merged["language_original"].astype(str).str.lower()
    != merged["language_translated"].astype(str).str.lower()
)

if language_mismatch.any():
    raise ValueError(
        f"Language mismatch for "
        f"{language_mismatch.sum()} examples."
    )


#analysis columsn

merged["gold_label"] = merged["gold_label_original"]

merged["language"] = (
    merged["language_original"]
    .astype(str)
    .str.strip()
    .str.lower()
)

merged["original_prediction"] = (
    merged["predicted_label_original"]
)

merged["translated_prediction"] = (
    merged["predicted_label_translated"]
)

merged["original_correct"] = (
    merged["original_prediction"]
    == merged["gold_label"]
)

merged["translated_correct"] = (
    merged["translated_prediction"]
    == merged["gold_label"]
)

# Transition categories

def transition_type(row):

    if (
        not row["original_correct"]
        and row["translated_correct"]
    ):
        return "wrong_to_correct"

    if (
        row["original_correct"]
        and not row["translated_correct"]
    ):
        return "correct_to_wrong"

    if (
        row["original_correct"]
        and row["translated_correct"]
    ):
        return "correct_to_correct"

    return "wrong_to_wrong"


merged["transition"] = merged.apply(
    transition_type,
    axis=1,
)


# Overall performance comparison

overall_rows = []

for experiment_name, pred_col in [
    ("Qwen 14B Original Gold Evidence", "original_prediction"),
    ("Qwen 14B Translated Gold Evidence", "translated_prediction"),
]:

    overall_rows.append({
        "Experiment": experiment_name,
        "Accuracy": accuracy_score(
            merged["gold_label"],
            merged[pred_col],
        ),
    })

overall_df = pd.DataFrame(overall_rows)

overall_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_accuracy_comparison.csv",
    ),
    index=False,
)


# Per-language analysis

language_rows = []

for language, group in merged.groupby("language"):

    original_accuracy = safe_accuracy(
        group,
        "original_prediction",
    )

    translated_accuracy = safe_accuracy(
        group,
        "translated_prediction",
    )

    language_rows.append({
        "Language": language,
        "Examples": len(group),
        "Original_Accuracy": original_accuracy,
        "Translated_Accuracy": translated_accuracy,
        "Absolute_Change": (
            translated_accuracy - original_accuracy
        ),
    })

language_df = pd.DataFrame(language_rows)

language_df = language_df.sort_values(
    "Absolute_Change",
    ascending=False,
)

language_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_language_accuracy.csv",
    ),
    index=False,
)

# Per-label analysis

label_rows = []

for label in LABELS:

    group = merged[
        merged["gold_label"] == label
    ]

    original_accuracy = safe_accuracy(
        group,
        "original_prediction",
    )

    translated_accuracy = safe_accuracy(
        group,
        "translated_prediction",
    )

    label_rows.append({
        "Gold_Label": label,
        "Examples": len(group),
        "Original_Accuracy": original_accuracy,
        "Translated_Accuracy": translated_accuracy,
        "Absolute_Change": (
            translated_accuracy - original_accuracy
        ),
    })

label_df = pd.DataFrame(label_rows)

label_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_label_accuracy.csv",
    ),
    index=False,
)


# Per-language AND per-label analysis

language_label_rows = []

for (language, label), group in merged.groupby(
    ["language", "gold_label"]
):

    original_accuracy = safe_accuracy(
        group,
        "original_prediction",
    )

    translated_accuracy = safe_accuracy(
        group,
        "translated_prediction",
    )

    language_label_rows.append({
        "Language": language,
        "Gold_Label": label,
        "Examples": len(group),
        "Original_Accuracy": original_accuracy,
        "Translated_Accuracy": translated_accuracy,
        "Absolute_Change": (
            translated_accuracy - original_accuracy
        ),
    })

language_label_df = pd.DataFrame(
    language_label_rows
)

language_label_df = language_label_df.sort_values(
    ["Language", "Gold_Label"]
)

language_label_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_language_per_label_accuracy.csv",
    ),
    index=False,
)


# Classification reports]

original_report = classification_report(
    merged["gold_label"],
    merged["original_prediction"],
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

translated_report = classification_report(
    merged["gold_label"],
    merged["translated_prediction"],
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

original_report_df = (
    pd.DataFrame(original_report)
    .transpose()
)

translated_report_df = (
    pd.DataFrame(translated_report)
    .transpose()
)

original_report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report_original.csv",
    )
)

translated_report_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report_translated.csv",
    )
)


# Confusion matrices

original_cm = confusion_matrix(
    merged["gold_label"],
    merged["original_prediction"],
    labels=LABELS,
)

translated_cm = confusion_matrix(
    merged["gold_label"],
    merged["translated_prediction"],
    labels=LABELS,
)

original_cm_df = pd.DataFrame(
    original_cm,
    index=[f"gold_{x}" for x in LABELS],
    columns=[f"pred_{x}" for x in LABELS],
)

translated_cm_df = pd.DataFrame(
    translated_cm,
    index=[f"gold_{x}" for x in LABELS],
    columns=[f"pred_{x}" for x in LABELS],
)

original_cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_original.csv",
    )
)

translated_cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix_translated.csv",
    )
)


# Transition summary

transition_summary = (
    merged["transition"]
    .value_counts()
    .rename_axis("Transition")
    .reset_index(name="Count")
)

transition_summary["Percentage"] = (
    transition_summary["Count"]
    / len(merged)
)

transition_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transition_summary.csv",
    ),
    index=False,
)


# Transition by language

transition_language = (
    merged.groupby(
        ["language", "transition"]
    )
    .size()
    .reset_index(name="Count")
)

transition_language.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transitions_by_language.csv",
    ),
    index=False,
)

# Transition by gold label

transition_label = (
    merged.groupby(
        ["gold_label", "transition"]
    )
    .size()
    .reset_index(name="Count")
)

transition_label.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transitions_by_label.csv",
    ),
    index=False,
)


# Save example-level transitions

wrong_to_correct = merged[
    merged["transition"] == "wrong_to_correct"
].copy()

correct_to_wrong = merged[
    merged["transition"] == "correct_to_wrong"
].copy()

wrong_to_wrong = merged[
    merged["transition"] == "wrong_to_wrong"
].copy()

wrong_to_correct.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wrong_to_correct_examples.csv",
    ),
    index=False,
)

correct_to_wrong.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "correct_to_wrong_examples.csv",
    ),
    index=False,
)

wrong_to_wrong.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "wrong_to_wrong_examples.csv",
    ),
    index=False,
)


merged.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "full_translation_comparison.csv",
    ),
    index=False,
)


#  key findings


print("\n" + "=" * 70)
print("OVERALL ACCURACY")
print("=" * 70)
print(overall_df.to_string(index=False))

print("\n" + "=" * 70)
print("PER-LANGUAGE ACCURACY")
print("=" * 70)
print(language_df.to_string(index=False))

print("\n" + "=" * 70)
print("PER-LABEL ACCURACY")
print("=" * 70)
print(label_df.to_string(index=False))

print("\n" + "=" * 70)
print("TRANSITION SUMMARY")
print("=" * 70)
print(transition_summary.to_string(index=False))

print("\n" + "=" * 70)
print("ORIGINAL CONFUSION MATRIX")
print("=" * 70)
print(original_cm_df)

print("\n" + "=" * 70)
print("TRANSLATED CONFUSION MATRIX")
print("=" * 70)
print(translated_cm_df)

print("\nAnalysis saved to:")
print(OUTPUT_DIR)