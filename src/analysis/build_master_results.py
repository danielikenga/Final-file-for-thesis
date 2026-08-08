import os
import json
import pandas as pd

OUTPUT_DIR = "results/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


EXPERIMENTS = [



    {
        "category": "Encoder",
        "name": "XLM-R Claim Only",
        "file": "results/baselines/xlmr_custom_split_results.json"
    },

    {
        "category": "Encoder",
        "name": "AfriBERTa Claim Only",
        "file": "results/baselines/afriberta_custom_split_results.json"
    },



    {
        "category": "Gold Evidence",
        "name": "XLM-R Gold Evidence",
        "file": "results/gold_evidence/xlmr_gold_evidence_results.json"
    },

    {
        "category": "Gold Evidence",
        "name": "AfriBERTa Gold Evidence",
        "file": "results/gold_evidence/afriberta_gold_evidence_results.json"
    },



    {
        "category": "LLM",
        "name": "Qwen 1.5B Claim Only",
        "file": "results/llm/qwen_claim_only_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 1.5B Gold Evidence",
        "file": "results/llm/qwen_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 1.5B Gold + Reasoning",
        "file": "results/llm/qwen_gold_evidence_reasoning_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 1.5B BM25 Evidence",
        "file": "results/llm/qwen_bm25_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 1.5B Adversarial Evidence",
        "file": "results/llm/qwen_adversarial_evidence_results.json"
    },



    {
        "category": "LLM",
        "name": "Qwen 14B Gold Evidence",
        "file": "results/llm/qwen14b_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 14B Few-shot (3)",
        "file": "results/llm/qwen14b_fewshot_3_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 14B Few-shot (6)",
        "file": "results/llm/qwen14b_fewshot_6_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 14B Few-shot (9)",
        "file": "results/llm/qwen14b_fewshot_9_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 14B + Google Translate",
        "file": "results/llm/qwen14b_translated_gold_evidence_results.json"
    },

    {
        "category": "LLM",
        "name": "Qwen 14B + Translate + Generic Examples",
        "file": "results/llm/qwen14b_translated_prompt_examples_results.json"
    },

]
def read_metrics(file_path):
    with open(file_path, "r") as f:
        result = json.load(f)

    if "accuracy" in result:
        return {
            "accuracy": result.get("accuracy"),
            "macro_f1": result.get("macro_f1"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "valid_predictions": result.get("valid_predictions"),
        }

    elif "eval_accuracy" in result:
        return {
            "accuracy": result.get("eval_accuracy"),
            "macro_f1": result.get("eval_macro_f1"),
            "precision": result.get("eval_precision"),
            "recall": result.get("eval_recall"),
            "valid_predictions": None,
        }

    else:
        raise ValueError(f"Unknown JSON format: {file_path}")




rows = []

for exp in EXPERIMENTS:

    metrics = read_metrics(exp["file"])

    rows.append({

        "Category": exp["category"],
        "Experiment": exp["name"],

        "Accuracy": metrics["accuracy"],
        "Macro_F1": metrics["macro_f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Valid Predictions": metrics["valid_predictions"]

    })

df = pd.DataFrame(rows)

df.to_csv(
    os.path.join(OUTPUT_DIR, "master_results.csv"),
    index=False
)

print(df)

print("\nSaved master_results.csv")
