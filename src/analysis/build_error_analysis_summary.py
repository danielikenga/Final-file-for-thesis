from pathlib import Path
import pandas as pd

CROSS_DIR = Path("results/analysis/cross_experiment_errors")
CASE_DIR = CROSS_DIR / "case_analysis"
OUTPUT_DIR = CROSS_DIR / "synthesis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

overall = pd.read_csv(CROSS_DIR / "overall_accuracy.csv")
pairwise = pd.read_csv(CROSS_DIR / "pairwise_transition_summary.csv")
per_language = pd.read_csv(CROSS_DIR / "per_language_accuracy.csv")
per_label = pd.read_csv(CROSS_DIR / "per_label_accuracy.csv")
prediction_distribution = pd.read_csv(
    CROSS_DIR / "prediction_distribution.csv"
)
reviewed = pd.read_csv(
    CASE_DIR / "persistent_hard_refutes_reviewed.csv"
)

# ---------------------------------------------------------
# 1. Overall model comparison
# ---------------------------------------------------------

overall_sorted = overall.sort_values(
    "Accuracy",
    ascending=False
).reset_index(drop=True)

overall_sorted["Rank"] = range(1, len(overall_sorted) + 1)

overall_sorted.to_csv(
    OUTPUT_DIR / "ranked_overall_accuracy.csv",
    index=False
)

# ---------------------------------------------------------
# 2. Key pairwise transitions
# ---------------------------------------------------------

key_pairs = [
    ("qwen_1_5b_claim_only", "qwen_1_5b_gold"),
    ("qwen_1_5b_gold", "qwen_1_5b_bm25"),
    ("qwen_1_5b_gold", "qwen_1_5b_adversarial"),
    ("qwen_1_5b_gold", "qwen_14b_gold"),
    ("qwen_14b_gold", "qwen_14b_translated"),
]

transition_rows = []

for before, after in key_pairs:
    subset = pairwise[
        (pairwise["Before"] == before) &
        (pairwise["After"] == after)
    ]

    counts = dict(
        zip(subset["Transition"], subset["Count"])
    )

    transition_rows.append({
        "Before": before,
        "After": after,
        "Correct_to_Correct": counts.get(
            "correct_to_correct", 0
        ),
        "Wrong_to_Correct": counts.get(
            "wrong_to_correct", 0
        ),
        "Correct_to_Wrong": counts.get(
            "correct_to_wrong", 0
        ),
        "Wrong_to_Wrong": counts.get(
            "wrong_to_wrong", 0
        ),
        "Net_Correct_Gain": (
            counts.get("wrong_to_correct", 0)
            - counts.get("correct_to_wrong", 0)
        ),
    })

transition_summary = pd.DataFrame(transition_rows)

transition_summary.to_csv(
    OUTPUT_DIR / "key_transition_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 3. Hard-refute taxonomy
# ---------------------------------------------------------

taxonomy_counts = (
    reviewed["primary_error_type"]
    .value_counts()
    .rename_axis("Primary_Error_Type")
    .reset_index(name="Count")
)

taxonomy_counts["Percentage"] = (
    taxonomy_counts["Count"] / len(reviewed)
)

taxonomy_counts.to_csv(
    OUTPUT_DIR / "hard_refute_taxonomy_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 4. Explicitness summary
# ---------------------------------------------------------

explicitness = (
    reviewed["contradiction_explicitness"]
    .value_counts()
    .rename_axis("Contradiction_Explicitness")
    .reset_index(name="Count")
)

explicitness["Percentage"] = (
    explicitness["Count"] / len(reviewed)
)

explicitness.to_csv(
    OUTPUT_DIR / "contradiction_explicitness_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 5. Annotation suspicion summary
# ---------------------------------------------------------

annotation = (
    reviewed["annotation_suspected"]
    .value_counts()
    .rename_axis("Annotation_Suspected")
    .reset_index(name="Count")
)

annotation["Percentage"] = (
    annotation["Count"] / len(reviewed)
)

annotation.to_csv(
    OUTPUT_DIR / "annotation_suspicion_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 6. Language x taxonomy
# ---------------------------------------------------------

language_taxonomy = pd.crosstab(
    reviewed["language"],
    reviewed["primary_error_type"]
)

language_taxonomy.to_csv(
    OUTPUT_DIR / "language_by_error_type.csv"
)

# ---------------------------------------------------------
# 7. Best model by language
# ---------------------------------------------------------

best_by_language = (
    per_language
    .sort_values(
        ["Language", "Accuracy"],
        ascending=[True, False]
    )
    .groupby("Language", as_index=False)
    .first()
)

best_by_language.to_csv(
    OUTPUT_DIR / "best_model_by_language.csv",
    index=False
)

# ---------------------------------------------------------
# 8. Best model by gold label
# ---------------------------------------------------------

best_by_label = (
    per_label
    .sort_values(
        ["Gold_Label", "Accuracy"],
        ascending=[True, False]
    )
    .groupby("Gold_Label", as_index=False)
    .first()
)

best_by_label.to_csv(
    OUTPUT_DIR / "best_model_by_label.csv",
    index=False
)

# ---------------------------------------------------------
# 9. Prediction bias summary
# ---------------------------------------------------------

bias_pivot = prediction_distribution.pivot(
    index="Experiment",
    columns="Predicted_Label",
    values="Percentage"
).reset_index()

bias_pivot.to_csv(
    OUTPUT_DIR / "prediction_bias_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 10. Human-readable synthesis
# ---------------------------------------------------------

best_row = overall_sorted.iloc[0]

n_hard = len(reviewed)

explicit_count = (
    reviewed["contradiction_explicitness"] == "explicit"
).sum()

implicit_count = (
    reviewed["contradiction_explicitness"] == "implicit"
).sum()

ambiguous_count = (
    reviewed["contradiction_explicitness"] == "ambiguous"
).sum()

annotation_yes = (
    reviewed["annotation_suspected"] == "yes"
).sum()

evidence_insufficient = (
    reviewed["primary_error_type"] == "evidence_insufficient"
).sum()

summary_lines = [
    "# Cross-Experiment Error Analysis Synthesis",
    "",
    "## Overall Performance",
    "",
    (
        f"The strongest evaluated system was "
        f"`{best_row['Experiment']}` with accuracy "
        f"{best_row['Accuracy']:.4f}."
    ),
    "",
    "## Persistent Hard Refutes",
    "",
    (
        f"A total of {n_hard} refutation examples were "
        f"misclassified by every analysed system."
    ),
    "",
    (
        f"- Explicit contradictions: {explicit_count}/{n_hard} "
        f"({explicit_count / n_hard:.1%})"
    ),
    (
        f"- Implicit contradictions: {implicit_count}/{n_hard} "
        f"({implicit_count / n_hard:.1%})"
    ),
    (
        f"- Ambiguous contradictions: {ambiguous_count}/{n_hard} "
        f"({ambiguous_count / n_hard:.1%})"
    ),
    (
        f"- Suspected annotation issues: {annotation_yes}/{n_hard} "
        f"({annotation_yes / n_hard:.1%})"
    ),
    (
        f"- Primarily evidence-insufficient cases: "
        f"{evidence_insufficient}/{n_hard} "
        f"({evidence_insufficient / n_hard:.1%})"
    ),
    "",
    "## Interpretation",
    "",
    (
        "Persistent errors should not be interpreted solely as "
        "model failures. The manually coded taxonomy indicates "
        "a mixture of genuine contradiction-reasoning failures, "
        "implicit semantic mismatches, and potentially ambiguous "
        "or evidence-insufficient benchmark instances."
    ),
    "",
    (
        "Translation improved aggregate performance but produced "
        "asymmetric effects across labels and languages. Earlier "
        "transition analysis showed particularly strong repair of "
        "refutation errors, alongside regressions for some support "
        "examples."
    ),
    "",
    (
        "Retrieval quality also materially affected classification. "
        "BM25 evidence introduced a strong support prediction bias, "
        "while adversarial evidence caused substantial correctness "
        "regressions, demonstrating that evidence availability alone "
        "is insufficient when relevance or reliability is poor."
    ),
    "",
    (
        "Model scaling from Qwen 1.5B to Qwen 14B repaired many "
        "previously incorrect examples, but did not eliminate "
        "persistent contradiction failures."
    ),
]

summary_path = OUTPUT_DIR / "error_analysis_synthesis.md"
summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)

# ---------------------------------------------------------
# Print core outputs
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RANKED OVERALL ACCURACY")
print("=" * 70)
print(
    overall_sorted[
        ["Rank", "Experiment", "Accuracy"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("KEY TRANSITION SUMMARY")
print("=" * 70)
print(transition_summary.to_string(index=False))

print("\n" + "=" * 70)
print("HARD REFUTE TAXONOMY")
print("=" * 70)
print(taxonomy_counts.to_string(index=False))

print("\n" + "=" * 70)
print("CONTRADICTION EXPLICITNESS")
print("=" * 70)
print(explicitness.to_string(index=False))

print("\n" + "=" * 70)
print("ANNOTATION SUSPICION")
print("=" * 70)
print(annotation.to_string(index=False))

print("\n" + "=" * 70)
print("BEST MODEL BY LANGUAGE")
print("=" * 70)
print(best_by_language.to_string(index=False))

print("\n" + "=" * 70)
print("BEST MODEL BY LABEL")
print("=" * 70)
print(best_by_label.to_string(index=False))

print("\nSaved synthesis outputs to:")
print(OUTPUT_DIR)