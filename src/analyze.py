import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_scored(path):
    df = pd.read_csv(path)
    for col in ("correct", "abstained", "hallucinated"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower() == "true"
    return df


def compute_summary(df):
    rows = []
    for cond in df["condition"].unique():
        subset = df[df["condition"] == cond]
        correct = subset["correct"]
        abstained = subset["abstained"]
        hallucinated = subset["hallucinated"]

        avg_conf_correct = subset.loc[subset["correct"], "confidence"].mean()
        avg_conf_incorrect = subset.loc[~subset["correct"], "confidence"].mean()

        rows.append({
            "condition": cond,
            "accuracy": correct.mean(),
            "abstain_rate": abstained.mean(),
            "hallucination_rate": hallucinated.mean(),
            "avg_conf_correct": avg_conf_correct if pd.notna(avg_conf_correct) else None,
            "avg_conf_incorrect": avg_conf_incorrect if pd.notna(avg_conf_incorrect) else None,
            "n": len(subset),
        })

    return pd.DataFrame(rows)


def plot_bar(ax, conditions, values, title, ylabel, color="steelblue"):
    x = range(len(conditions))
    ax.bar(x, values, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=PROJECT_ROOT / "results" / "scored.csv",
        type=Path,
        help="Input scored CSV",
    )
    parser.add_argument(
        "--summary",
        default=PROJECT_ROOT / "results" / "summary.csv",
        type=Path,
        help="Output summary CSV",
    )
    parser.add_argument(
        "--plots-dir",
        default=PROJECT_ROOT / "results" / "plots",
        type=Path,
        help="Directory for plot PNGs",
    )
    args = parser.parse_args()

    df = load_scored(args.input)
    summary = compute_summary(df)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"Summary saved: {args.summary}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    conditions = summary["condition"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_bar(axes[0], conditions, summary["accuracy"].tolist(), "Accuracy by Condition", "Accuracy", "steelblue")
    plot_bar(axes[1], conditions, summary["hallucination_rate"].tolist(), "Hallucination Rate by Condition", "Hallucination Rate", "coral")
    plot_bar(axes[2], conditions, summary["abstain_rate"].tolist(), "Abstain Rate by Condition", "Abstain Rate", "seagreen")

    plt.tight_layout()
    acc_path = args.plots_dir / "accuracy_by_condition.png"
    hall_path = args.plots_dir / "hallucination_rate_by_condition.png"
    abst_path = args.plots_dir / "abstain_rate_by_condition.png"

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plot_bar(ax1, conditions, summary["accuracy"].tolist(), "Accuracy by Condition", "Accuracy", "steelblue")
    plt.tight_layout()
    fig1.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_bar(ax2, conditions, summary["hallucination_rate"].tolist(), "Hallucination Rate by Condition", "Hallucination Rate", "coral")
    plt.tight_layout()
    fig2.savefig(hall_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    plot_bar(ax3, conditions, summary["abstain_rate"].tolist(), "Abstain Rate by Condition", "Abstain Rate", "seagreen")
    plt.tight_layout()
    fig3.savefig(abst_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"Plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
