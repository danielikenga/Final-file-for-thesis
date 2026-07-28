import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INPUT_PATH = "data/processed/afrifact_nigerian_languages.jsonl"
OUTPUT_PATH = "data/processed/afrifact_nigerian_languages_custom_split.jsonl"


def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    df["stratify_key"] = df["language"] + "_" + df["label"]

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["stratify_key"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=2 / 3,
        random_state=SEED,
        stratify=temp_df["stratify_key"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "custom_train"
    val_df["split"] = "custom_validation"
    test_df["split"] = "custom_test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df = final_df.drop(columns=["stratify_key"])

    final_df.to_json(OUTPUT_PATH, orient="records", lines=True, force_ascii=False)

    print("Custom split saved to:", OUTPUT_PATH)
    print(final_df[["language", "split", "label"]].value_counts().sort_index())
    print("\nSplit sizes:")
    print(final_df["split"].value_counts())


if __name__ == "__main__":
    main()