import os
import pandas as pd

INPUT_PATH = (
    "results/analysis/cross_experiment_errors/"
    "merged_cross_experiment_predictions.csv"
)

OUTPUT_DIR = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis"
)

SOURCE_DATA_PATH = (
    "data/processed/"
    "afrifact_nigerian_languages_custom_split.jsonl"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def shorten(text, n=500):
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


print("Loading merged predictions...")
df = pd.read_csv(INPUT_PATH)

print("Loading source dataset...")
source = pd.read_json(SOURCE_DATA_PATH, lines=True)

source = source[
    source["split"] == "custom_test"
].copy()

source = source[
    [
        "id",
        "claim",
        "extracted_evidence_text",
        "label",
        "language",
    ]
].copy()

source = source.rename(
    columns={
        "claim": "source_claim",
        "extracted_evidence_text": "source_gold_evidence",
        "label": "source_gold_label",
        "language": "source_language",
    }
)

df = df.merge(
    source,
    on="id",
    how="left",
)

correct_cols = [
    c for c in df.columns
    if c.endswith("_correct")
]

prediction_cols = [
    c for c in df.columns
    if c.endswith("_prediction")
]

df["num_systems_correct"] = df[correct_cols].sum(axis=1)


hard = df[
    df["num_systems_correct"] == 0
].copy()

hard = hard.sort_values(
    ["gold_label", "language", "id"]
)

hard.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_universally_hard_examples.csv"
    ),
    index=False,
)



hard_refutes = hard[
    hard["gold_label"] == "refutes"
].copy()

hard_refutes.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "persistent_hard_refutes.csv"
    ),
    index=False,
)


bm25_support_bias = df[
    (df["gold_label"] != "supports")
    & (df["qwen_1_5b_bm25_prediction"] == "supports")
].copy()

bm25_support_bias.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "bm25_support_bias_failures.csv"
    ),
    index=False,
)


scale_repairs = df[
    (~df["qwen_1_5b_gold_correct"])
    & (df["qwen_14b_gold_correct"])
].copy()

scale_repairs.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "scale_repairs_1_5b_to_14b.csv"
    ),
    index=False,
)


scale_regressions = df[
    (df["qwen_1_5b_gold_correct"])
    & (~df["qwen_14b_gold_correct"])
].copy()

scale_regressions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "scale_regressions_1_5b_to_14b.csv"
    ),
    index=False,
)


retrieval_damage = df[
    (df["qwen_1_5b_gold_correct"])
    & (~df["qwen_1_5b_bm25_correct"])
].copy()

retrieval_damage.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "retrieval_damage_gold_to_bm25.csv"
    ),
    index=False,
)


adversarial_damage = df[
    (df["qwen_1_5b_gold_correct"])
    & (~df["qwen_1_5b_adversarial_correct"])
].copy()

adversarial_damage.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "adversarial_damage.csv"
    ),
    index=False,
)


print("\nCASE COUNTS")
print("Universally hard:", len(hard))
print("Persistent hard refutes:", len(hard_refutes))
print("BM25 support-bias failures:", len(bm25_support_bias))
print("Scale repairs 1.5B -> 14B:", len(scale_repairs))
print("Scale regressions 1.5B -> 14B:", len(scale_regressions))
print("Retrieval damage gold -> BM25:", len(retrieval_damage))
print("Adversarial damage:", len(adversarial_damage))

print("\nPERSISTENT HARD REFUTES BY LANGUAGE")
print(hard_refutes["language"].value_counts().to_string())

print("\nBM25 SUPPORT-BIAS FAILURES BY GOLD LABEL")
print(bm25_support_bias["gold_label"].value_counts().to_string())

print("\nBM25 SUPPORT-BIAS FAILURES BY LANGUAGE")
print(bm25_support_bias["language"].value_counts().to_string())

print("\nSAMPLE UNIVERSALLY HARD REFUTES")

for _, row in hard_refutes.head(8).iterrows():
    print("\n" + "=" * 70)
    print("ID:", row["id"])
    print("Language:", row["language"])
    print("Gold:", row["gold_label"])
    print("Claim:", shorten(row["source_claim"]))
    print("Evidence:", shorten(row["source_gold_evidence"]))

    print("\nPredictions:")
    for col in prediction_cols:
        print(
            f"  {col.replace('_prediction', '')}: "
            f"{row[col]}"
        )

print("\nSaved case analysis to:")
print(OUTPUT_DIR)