import os
import pandas as pd
import matplotlib.pyplot as plt

MASTER_RESULTS = "results/analysis/master_results.csv"

OUTPUT_DIR = "results/analysis/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(MASTER_RESULTS)



# Helper plotting function


def save_bar_plot(
    data,
    x_col,
    y_col,
    title,
    ylabel,
    filename,
    color="#4C72B0"
):

    plt.figure(figsize=(8,5))

    bars = plt.bar(
        data[x_col],
        data[y_col],
        color=color
    )

    plt.ylabel(ylabel)

    plt.title(title)

    plt.xticks(rotation=20, ha="right")

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()

# Encoder Comparison


encoder_df = df[
    df["Category"].isin(
        ["Encoder", "Gold Evidence"]
    )
].copy()

save_bar_plot(

    encoder_df,

    "Experiment",

    "Accuracy",

    "Encoder Model Comparison",

    "Accuracy",

    "encoder_accuracy.png"

)

save_bar_plot(

    encoder_df,

    "Experiment",

    "Macro_F1",

    "Encoder Model Comparison (Macro F1)",

    "Macro F1",

    "encoder_macrof1.png"

)
print("Encoder figures generated successfully.")