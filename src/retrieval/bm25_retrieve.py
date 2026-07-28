import os
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


DATA_PATH = "data/processed/afrifact_nigerian_languages_custom_split.jsonl"
OUTPUT_PATH = "results/retrieval/bm25_top10_retrieved_evidence.csv"
METRICS_PATH = "results/retrieval/bm25_retrieval_metrics.json"


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def main():
    os.makedirs("results/retrieval", exist_ok=True)

    df = pd.read_json(DATA_PATH, lines=True)

    corpus_df = df.copy()
    corpus_df["extracted_evidence_text"] = (
        corpus_df["extracted_evidence_text"]
        .fillna("")
        .astype(str)
    )

    corpus_texts = corpus_df["extracted_evidence_text"].tolist()
    tokenized_corpus = [tokenize(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    test_df = df[df["split"] == "custom_test"].copy()

    rows = []
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    recall_at_10 = []

    for count, (_, row) in enumerate(test_df.iterrows(), start=1):
        claim = str(row["claim"])
        query_tokens = tokenize(claim)

        scores = bm25.get_scores(query_tokens)

        top_k_indices = np.argsort(scores)[::-1][:10]
        top_k_ids = [corpus_df.iloc[idx]["id"] for idx in top_k_indices]

        gold_id = row["id"]

        recall_at_1.append(gold_id in top_k_ids[:1])
        recall_at_3.append(gold_id in top_k_ids[:3])
        recall_at_5.append(gold_id in top_k_ids[:5])
        recall_at_10.append(gold_id in top_k_ids[:10])

        for rank, idx in enumerate(top_k_indices, start=1):
            retrieved_row = corpus_df.iloc[idx]

            rows.append({
                "query_id": gold_id,
                "language": row["language"],
                "claim": claim,
                "gold_label": row["label"],
                "gold_evidence": row["extracted_evidence_text"],
                "rank": rank,
                "retrieved_id": retrieved_row["id"],
                "retrieved_language": retrieved_row["language"],
                "retrieved_evidence": retrieved_row["extracted_evidence_text"],
                "bm25_score": scores[idx],
                "is_gold_evidence": gold_id == retrieved_row["id"],
            })

        print(
            f"{count}/{len(test_df)} | "
            f"R@1={recall_at_1[-1]} "
            f"R@3={recall_at_3[-1]} "
            f"R@5={recall_at_5[-1]} "
            f"R@10={recall_at_10[-1]}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)

    metrics = {
        "recall_at_1": float(np.mean(recall_at_1)),
        "recall_at_3": float(np.mean(recall_at_3)),
        "recall_at_5": float(np.mean(recall_at_5)),
        "recall_at_10": float(np.mean(recall_at_10)),
        "total_queries": int(len(test_df)),
    }

    print("\nBM25 RETRIEVAL METRICS")
    print(metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_json(METRICS_PATH, orient="records", indent=4)

    print(f"\nSaved top-10 retrievals to {OUTPUT_PATH}")
    print(f"Saved retrieval metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()