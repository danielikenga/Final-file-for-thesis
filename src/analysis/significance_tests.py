"""significance testing for the key experimental comparisons.
For two systems A and B evaluated on the same n examples:
    b = number of examples A got right and B got wrong
    c = number of examples A got wrong and B got right
Under the null hypothesis that A and B have equal error rates, b is
Binomial(b + c, 0.5)... I report a two-sided exact p-value.

"""

import pandas as pd
from scipy.stats import binomtest

RESULTS_DIR = "results/llm"
OUTPUT_PATH = "results/analysis/significance_tests.csv"


def load_correct(path, id_col="id"):
    df = pd.read_csv(path)
    df = df[[id_col, "correct"]].rename(columns={id_col: "id"})
    df["correct"] = df["correct"].astype(bool)
    return df.set_index("id")["correct"]


def mcnemar_exact(correct_a, correct_b, label_a, label_b):
    """Exact two-sided McNemar's test on matched correctness pairs"""
    joined = pd.concat([correct_a, correct_b], axis=1, join="inner")
    joined.columns = ["a", "b"]

    n_matched = len(joined)

    b = int(((joined["a"] == True) & (joined["b"] == False)).sum())
    c = int(((joined["a"] == False) & (joined["b"] == True)).sum())

    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(min(b, c), discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_matched": n_matched,
        f"{label_a}_correct": int(joined["a"].sum()),
        f"{label_b}_correct": int(joined["b"].sum()),
        "a_right_b_wrong (b)": b,
        "a_wrong_b_right (c)": c,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def main():
    conditions = {
        "qwen1.5b_claim_only": load_correct(f"{RESULTS_DIR}/qwen_claim_only_predictions.csv"),
        "qwen1.5b_gold": load_correct(f"{RESULTS_DIR}/qwen_gold_evidence_predictions.csv"),
        "qwen1.5b_bm25": load_correct(f"{RESULTS_DIR}/qwen_bm25_evidence_predictions.csv", id_col="query_id"),
        "qwen1.5b_adversarial": load_correct(f"{RESULTS_DIR}/qwen_adversarial_evidence_predictions.csv"),
        "qwen14b_gold": load_correct(f"{RESULTS_DIR}/qwen14b_gold_evidence_predictions.csv"),
        "qwen14b_translated_gold": load_correct(f"{RESULTS_DIR}/qwen14b_translated_gold_evidence_predictions.csv"),
    }

    comparisons = [
        # Evidence availability
        ("qwen1.5b_claim_only", "qwen1.5b_gold"),
        # Oracle vs retrieved evidence
        ("qwen1.5b_gold", "qwen1.5b_bm25"),
        # Robustness to misleading evidence
        ("qwen1.5b_gold", "qwen1.5b_adversarial"),
        ("qwen1.5b_claim_only", "qwen1.5b_adversarial"),
        # Model scale ( confounded with 4-bit quantization at 14B
        ("qwen1.5b_gold", "qwen14b_gold"),
        # Translation effect (same model + quantization, only input language changes)
        ("qwen14b_gold", "qwen14b_translated_gold"),
    ]

    rows = []
    for a_label, b_label in comparisons:
        rows.append(
            mcnemar_exact(conditions[a_label], conditions[b_label], a_label, b_label)
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(results_df)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(
        "\nCAVEAT: the qwen1.5b_gold vs qwen14b_gold comparison is confounded by "
        "quantization (1.5B run in full precision, 14B run in 4-bit NF4) as well as "
        "model scale. Significance here should be reported as 'the two configurations "
        "differ significantly' rather than attributed purely to parameter count."
    )


if __name__ == "__main__":
    main()
