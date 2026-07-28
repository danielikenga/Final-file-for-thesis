import json
import os
import re
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/processed/afrifact_nigerian_languages_custom_split.jsonl"

NUM_SHOTS = 3
RESULTS_PATH = f"results/llm/qwen_fewshot_{NUM_SHOTS}_gold_evidence_results.json"
PREDICTIONS_PATH = f"results/llm/qwen_fewshot_{NUM_SHOTS}_gold_evidence_predictions.csv"

LABELS = ["supports", "refutes", "nei"]


def build_fewshot_examples(train_df, num_shots):
    if num_shots == 0:
        return ""

    if num_shots % 3 != 0:
        raise ValueError("NUM_SHOTS should be 0, 3, 6, or 9 for balanced labels.")

    examples = []
    labels_cycle = ["supports", "refutes", "nei"]
    shots_per_label = num_shots // 3

    for label in labels_cycle:
        label_rows = train_df[
            (train_df["label"] == label) &
            (train_df["extracted_evidence_text"].notna())
        ]

        sampled_rows = label_rows.sample(
            n=shots_per_label,
            random_state=42,
            replace=False
        )

        for _, sampled in sampled_rows.iterrows():
            examples.append(f"""
Example {len(examples) + 1}

Evidence:
{sampled["extracted_evidence_text"]}

Claim:
{sampled["claim"]}

Answer:
{sampled["label"]}
""")

    return "\n".join(examples)


def build_prompt(claim, evidence, fewshot_examples):
    return f"""
You are a strict fact verification system.

Your task is to classify the claim using ONLY the evidence provided.

Choose exactly one label:
- supports
- refutes
- nei

Definitions:
supports = the evidence supports the claim.
refutes = the evidence contradicts the claim.
nei = the evidence does not provide enough information to verify the claim.

Here are some of the examples of the task:

{fewshot_examples}

Now classify the following claim.

Evidence:
{evidence}

Claim:
{claim}

Return only one word: supports, refutes, or nei.
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
    train_df = df[df["split"] == "custom_train"].copy()
    fewshot_examples = build_fewshot_examples(train_df, NUM_SHOTS)

    LIMIT = 20
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

        prompt = build_prompt(claim, evidence, fewshot_examples)

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

    results["num_shots"] = NUM_SHOTS

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"\nSaved results to {RESULTS_PATH}")
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()