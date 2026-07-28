import json
import os
import re
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/processed/afrifact_nigerian_languages_custom_split.jsonl"

RESULTS_PATH = "results/llm/qwen_gold_evidence_reasoning_results.json"
PREDICTIONS_PATH = "results/llm/qwen_gold_evidence_reasoning_predictions.csv"

LABELS = ["supports", "refutes", "nei"]


def build_prompt(claim, evidence):
    return f"""
You are a professional fact-checking system.

Your task is to determine whether the evidence supports,
refutes, or provides insufficient information for the claim.

Reason carefully using ONLY the evidence provided.

Evidence:
{evidence}

Claim:
{claim}

First decide:

1. Does the evidence support the claim?
2. Does the evidence contradict the claim?
3. Is there not enough information?

After reasoning, output ONLY ONE LABEL:

supports
refutes
nei
"""
def extract_label(response_text):
    text = response_text.lower().strip()

    matches = re.findall(r"\b(supports|refutes|nei)\b", text)

    if len(matches) > 0:
        return matches[-1]

    if "not enough" in text:
        return "nei"

    return "unknown"


def compute_metrics(y_true, y_pred):
    filtered_true = []
    filtered_pred = []

    for true, pred in zip(y_true, y_pred):
        if pred in LABELS:
            filtered_true.append(true)
            filtered_pred.append(pred)

    accuracy = accuracy_score(filtered_true, filtered_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_true,
        filtered_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        "valid_predictions": len(filtered_pred),
        "total_predictions": len(y_pred),
        "invalid_predictions": len(y_pred) - len(filtered_pred),
    }


def main():
    os.makedirs("results/llm", exist_ok=True)

    df = pd.read_json(DATA_PATH, lines=True)
    test_df = df[df["split"] == "custom_test"].copy()


    LIMIT = None
    if LIMIT is not None:
        test_df = test_df.head(LIMIT)

    print("Test examples:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    predictions = []

    for idx, row in test_df.iterrows():
        claim = str(row["claim"])
        evidence = str(row["extracted_evidence_text"]) if pd.notna(row["extracted_evidence_text"]) else ""
        gold_label = str(row["label"])

        prompt = build_prompt(claim, evidence)

        messages = [
            {"role": "system", "content": "You are a strict fact verification classifier."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        predicted_label = extract_label(response_text)

        predictions.append({
            "id": row["id"],
            "language": row["language"],
            "claim": claim,
            "gold_evidence": evidence,
            "gold_label": gold_label,
            "raw_response": response_text,
            "predicted_label": predicted_label,
            "correct": predicted_label == gold_label,
        })

        print(
            f"{len(predictions)}/{len(test_df)} | "
            f"gold={gold_label} | pred={predicted_label} | response={response_text.strip()}"
        )

    prediction_df = pd.DataFrame(predictions)

    results = compute_metrics(
        prediction_df["gold_label"].tolist(),
        prediction_df["predicted_label"].tolist(),
    )

    print("\nRESULTS")
    print(results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nSaved results to {RESULTS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()