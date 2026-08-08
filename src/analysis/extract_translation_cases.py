import os
import pandas as pd

INPUT_PATH = (
    "results/analysis/translation_effect/"
    "full_translation_comparison.csv"
)

OUTPUT_DIR = (
    "results/analysis/translation_effect/"
    "case_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

print("Loaded examples:", len(df))




def first_existing_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


original_claim_col = first_existing_column(
    df,
    [
        "claim_original",
        "claim",
    ],
)

translated_claim_col = first_existing_column(
    df,
    [
        "claim_translated",
        "translated_claim",
    ],
)

original_evidence_col = first_existing_column(
    df,
    [
        "gold_evidence_original",
        "evidence_original",
        "gold_evidence",
        "evidence",
    ],
)

translated_evidence_col = first_existing_column(
    df,
    [
        "gold_evidence_translated",
        "evidence_translated",
        "translated_evidence",
    ],
)

print("Original claim column:", original_claim_col)
print("Translated claim column:", translated_claim_col)
print("Original evidence column:", original_evidence_col)
print("Translated evidence column:", translated_evidence_col)




if original_claim_col is not None:
    df["original_claim_length"] = (
        df[original_claim_col]
        .fillna("")
        .astype(str)
        .str.len()
    )
else:
    df["original_claim_length"] = 0

if original_evidence_col is not None:
    df["original_evidence_length"] = (
        df[original_evidence_col]
        .fillna("")
        .astype(str)
        .str.len()
    )
else:
    df["original_evidence_length"] = 0

df["total_original_text_length"] = (
    df["original_claim_length"]
    + df["original_evidence_length"]
)


# Target group 1:
# Refutes cases repaired by translation


repaired_refutes = df[
    (df["gold_label"] == "refutes")
    & (df["transition"] == "wrong_to_correct")
].copy()

print(
    "\nRefutes wrong -> correct:",
    len(repaired_refutes),
)


# Target group 2:
# Yoruba supports damaged by translation


damaged_yoruba_supports = df[
    (df["language"] == "yoruba")
    & (df["gold_label"] == "supports")
    & (df["transition"] == "correct_to_wrong")
].copy()

print(
    "Yoruba supports correct -> wrong:",
    len(damaged_yoruba_supports),
)


# Saveing  ALL the  target cases


repaired_refutes.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_repaired_refutes.csv",
    ),
    index=False,
)

damaged_yoruba_supports.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_damaged_yoruba_supports.csv",
    ),
    index=False,
)



# Deterministic representative selection
# I am selecting examples around text-length quantiles:
# short, medium-short, median, medium-long, long.
# This avoids manually cherry-picking examples.


def select_quantile_cases(group, n=5):

    if len(group) == 0:
        return group.copy()

    ordered = group.sort_values(
        [
            "total_original_text_length",
            "id",
        ]
    ).reset_index(drop=True)

    if len(ordered) <= n:
        return ordered.copy()

    quantiles = [
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    ]

    selected_indices = []

    for q in quantiles:
        idx = round(
            q * (len(ordered) - 1)
        )
        selected_indices.append(idx)

    selected_indices = list(
        dict.fromkeys(selected_indices)
    )

    selected = ordered.iloc[
        selected_indices
    ].copy()

    return selected


representative_repaired_refutes = (
    select_quantile_cases(
        repaired_refutes,
        n=5,
    )
)

representative_damaged_yoruba_supports = (
    select_quantile_cases(
        damaged_yoruba_supports,
        n=5,
    )
)




representative_repaired_refutes.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "representative_repaired_refutes.csv",
    ),
    index=False,
)

representative_damaged_yoruba_supports.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "representative_damaged_yoruba_supports.csv",
    ),
    index=False,
)




review_columns = [
    "id",
    "language",
    "gold_label",
    "original_prediction",
    "translated_prediction",
    "transition",
]

for column in [
    original_claim_col,
    translated_claim_col,
    original_evidence_col,
    translated_evidence_col,
]:
    if (
        column is not None
        and column not in review_columns
    ):
        review_columns.append(column)


repaired_review = (
    representative_repaired_refutes[
        review_columns
    ].copy()
)

damaged_review = (
    representative_damaged_yoruba_supports[
        review_columns
    ].copy()
)


repaired_review.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "review_repaired_refutes.csv",
    ),
    index=False,
)

damaged_review.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "review_damaged_yoruba_supports.csv",
    ),
    index=False,
)




print("\n" + "=" * 70)
print("REPRESENTATIVE REPAIRED REFUTES")
print("=" * 70)

for _, row in repaired_review.iterrows():

    print("\nID:", row["id"])
    print("Language:", row["language"])
    print(
        "Prediction:",
        row["original_prediction"],
        "->",
        row["translated_prediction"],
    )

    if original_claim_col is not None:
        print(
            "Original claim:",
            str(row[original_claim_col])[:500],
        )

    if translated_claim_col is not None:
        print(
            "Translated claim:",
            str(row[translated_claim_col])[:500],
        )

    if original_evidence_col is not None:
        print(
            "Original evidence:",
            str(row[original_evidence_col])[:700],
        )

    if translated_evidence_col is not None:
        print(
            "Translated evidence:",
            str(row[translated_evidence_col])[:700],
        )


print("\n" + "=" * 70)
print("REPRESENTATIVE DAMAGED YORUBA SUPPORTS")
print("=" * 70)

for _, row in damaged_review.iterrows():

    print("\nID:", row["id"])
    print("Language:", row["language"])
    print(
        "Prediction:",
        row["original_prediction"],
        "->",
        row["translated_prediction"],
    )

    if original_claim_col is not None:
        print(
            "Original claim:",
            str(row[original_claim_col])[:500],
        )

    if translated_claim_col is not None:
        print(
            "Translated claim:",
            str(row[translated_claim_col])[:500],
        )

    if original_evidence_col is not None:
        print(
            "Original evidence:",
            str(row[original_evidence_col])[:700],
        )

    if translated_evidence_col is not None:
        print(
            "Translated evidence:",
            str(row[translated_evidence_col])[:700],
        )


print("\nCase analysis saved to:")
print(OUTPUT_DIR)