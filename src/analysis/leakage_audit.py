import pandas as pd
from collections import Counter
import re

df = pd.read_json(
    "data/processed/afrifact_nigerian_languages.jsonl",
    lines=True
)

for label in ["supports", "refutes", "nei"]:

    text = " ".join(
        df[df["label"] == label]
        ["extracted_evidence_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    words = re.findall(r"\w+", text.lower())

    counter = Counter(words)

    print("\n")
    print("=" * 50)
    print(label.upper())
    print("=" * 50)

    for word, count in counter.most_common(30):
        print(word, count)
