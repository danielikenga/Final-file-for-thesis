import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm
import os
import json

INPUT_PATH = "data/processed/afrifact_nigerian_languages_custom_split.jsonl"
OUTPUT_PATH = "data/processed/afrifact_translated_test.jsonl"
FALLBACK_LOG_PATH = "data/processed/translation_fallback_log.json"

# Explicit ISO-639-1 source codes per project language, to protect against misclassificatio of language
LANGUAGE_CODES = {
    "igbo": "ig",
    "yoruba": "yo",
    "hausa": "ha",
}


def translate_text(text, source_lang):

    if pd.isna(text):
        return "", False

    text = str(text).strip()

    if text == "":
        return "", False

    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text), False
    except Exception as e:
        print("Translation failed, falling back to source text:", e)
        return text, True


def main():

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_json(INPUT_PATH, lines=True)

    test_df = df[df["split"] == "custom_test"].copy()
    #test_df = test_df.head(5) (testing translation is stable with 5)

    print(f"Translating {len(test_df)} examples...")

    translated_claims = []
    translated_evidence = []
    claim_fallback_flags = []
    evidence_fallback_flags = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

        source_lang = LANGUAGE_CODES.get(str(row["language"]).lower())
        if source_lang is None:
            raise ValueError(f"Unknown language for row: {row['language']}")

        claim_text, claim_fell_back = translate_text(row["claim"], source_lang)
        evidence_text, evidence_fell_back = translate_text(
            row["extracted_evidence_text"], source_lang
        )

        translated_claims.append(claim_text)
        translated_evidence.append(evidence_text)
        claim_fallback_flags.append(claim_fell_back)
        evidence_fallback_flags.append(evidence_fell_back)

    test_df["translated_claim"] = translated_claims
    test_df["translated_evidence"] = translated_evidence
    test_df["claim_translation_fallback"] = claim_fallback_flags
    test_df["evidence_translation_fallback"] = evidence_fallback_flags

    test_df.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False
    )

    n = len(test_df)
    fallback_summary = {
        "total_examples": n,
        "claim_fallback_count": int(sum(claim_fallback_flags)),
        "claim_fallback_rate": sum(claim_fallback_flags) / n,
        "evidence_fallback_count": int(sum(evidence_fallback_flags)),
        "evidence_fallback_rate": sum(evidence_fallback_flags) / n,
    }

    with open(FALLBACK_LOG_PATH, "w") as f:
        json.dump(fallback_summary, f, indent=4)

    print("\nTranslation fallback summary (report this in the dissertation):")
    print(fallback_summary)

    print("\nSaved translated dataset to:")
    print(OUTPUT_PATH)
    print(f"Saved fallback log to {FALLBACK_LOG_PATH}")


if __name__ == "__main__":
    main()