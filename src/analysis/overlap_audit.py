import pandas as pd

def overlap_ratio(claim, evidence):

    claim_words = set(str(claim).lower().split())
    evidence_words = set(str(evidence).lower().split())

    if len(claim_words) == 0:
        return 0

    return len(
        claim_words.intersection(evidence_words)
    ) / len(claim_words)

df = pd.read_json(
    "data/processed/afrifact_nigerian_languages.jsonl",
    lines=True
)

df["overlap"] = df.apply(
    lambda row:
    overlap_ratio(
        row["claim"],
        row["extracted_evidence_text"]
    ),
    axis=1
)

print(
    df.groupby("label")["overlap"]
    .agg(["mean","median","std"])
)
