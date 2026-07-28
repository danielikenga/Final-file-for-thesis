from pathlib import Path
from typing import List
import pandas as pd
from datasets import load_dataset
LANGUAGE_CONFIGS = {
    "igbo": "igbo_culture_news",
    "yoruba": "yoruba_culture_news",
    "hausa": "hausa_culture_news",
}

SPLITS = ["train", "validation", "test", "fewshot"]

def normalise_label(label: str) -> str:
    label = str(label).strip()
    label_map = {
        "supports": "supports",
        "refutes": "refutes",
        "Not_Enough_Information": "nei",
        "not_enough_information": "nei",
        "NEI": "nei",
    }
    return label_map.get(label, label.lower())
def load_language_split(language: str, config_name: str, split: str) -> pd.DataFrame:
    dataset = load_dataset("masakhane/AfrIFact", config_name, split=split)
    df = dataset.to_pandas()

    df["language"] = language
    df["config"] = config_name
    df["split"] = split
    df["label"] = df["class"].apply(normalise_label)

    return df


def load_selected_afrifact() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for language, config_name in LANGUAGE_CONFIGS.items():
        for split in SPLITS:
            print(f"Loading {config_name} / {split}")
            df = load_language_split(language, config_name, split)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def save_processed_dataset(df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Saved processed dataset to {output_file}")
    print(f"Rows: {len(df)}")


def main():
    df = load_selected_afrifact()

    print("\n The dataset has loaded successfully")
    print(df[["language", "split", "label"]].value_counts().sort_index())

    save_processed_dataset(df, "data/processed/afrifact_nigerian_languages.jsonl")


if __name__ == "__main__":
    main()