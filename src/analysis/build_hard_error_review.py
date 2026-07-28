import os
import pandas as pd

INPUT_PATH = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis/persistent_hard_refutes.csv"
)

OUTPUT_DIR = (
    "results/analysis/cross_experiment_errors/"
    "case_analysis"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "persistent_hard_refutes_manual_review.csv"
)

df = pd.read_csv(INPUT_PATH)

review = pd.DataFrame({
    "id": df["id"],
    "language": df["language"],
    "gold_label": df["gold_label"],
    "claim": df["source_claim"],
    "gold_evidence": df["source_gold_evidence"],

    # Manual taxonomy fields
    "primary_error_type": "",
    "secondary_error_type": "",
    "contradiction_explicitness": "",
    "requires_world_knowledge": "",
    "translation_issue_present": "",
    "annotation_suspected": "",
    "review_notes": "",
})

review.to_csv(OUTPUT_PATH, index=False)

print("Created manual review file:")
print(OUTPUT_PATH)
print("Rows:", len(review))

print("\nSuggested primary_error_type categories:")
print("- negation")
print("- temporal_mismatch")
print("- numerical_mismatch")
print("- entity_attribute_mismatch")
print("- causal_mismatch")
print("- role_relation_mismatch")
print("- boundary_condition")
print("- implicit_contradiction")
print("- evidence_insufficient")
print("- possible_annotation_issue")
print("- other")

print("\nSuggested contradiction_explicitness values:")
print("- explicit")
print("- implicit")
print("- ambiguous")

print("\nBoolean fields:")
print("- yes")
print("- no")
print("- uncertain")