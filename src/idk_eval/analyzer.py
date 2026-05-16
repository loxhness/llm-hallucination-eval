import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idk_eval.paths import project_root

# Palette for multi-model charts; supports up to 6 models.
_MODEL_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


def load_scored(path):
    df = pd.read_csv(path)
    for col in ("correct", "abstained", "hallucinated"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower() == "true"
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (condition, model_name) with accuracy / rate / Brier stats."""
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = "unknown"

    rows = []
    for (cond, model), subset in df.groupby(["condition", "model_name"], sort=False):
        avg_conf_correct = subset.loc[subset["correct"], "confidence"].mean()
        avg_conf_incorrect = subset.loc[~subset["correct"], "confidence"].mean()

        # Brier score: mean((predicted_prob - outcome)^2), confidence normalised to [0, 1].
        # Rows missing confidence are excluded from this stat only.
        conf = pd.to_numeric(subset["confidence"], errors="coerce").dropna() / 100
        brier = (
            float(((conf - subset.loc[conf.index, "correct"].astype(float)) ** 2).mean())
            if not conf.empty else None
        )

        rows.append({
            "condition": cond,
            "model_name": model,
            "accuracy": subset["correct"].mean(),
            "abstain_rate": subset["abstained"].mean(),
            "hallucination_rate": subset["hallucinated"].mean(),
            "avg_conf_correct": avg_conf_correct if pd.notna(avg_conf_correct) else None,
            "avg_conf_incorrect": avg_conf_incorrect if pd.notna(avg_conf_incorrect) else None,
            "brier_score": brier,
            "n": len(subset),
        })
    return pd.DataFrame(rows)


def compute_calibration_data(df: pd.DataFrame, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """
    Bucket each model's predictions by confidence level, compute actual accuracy per bucket.

    Aggregates across all conditions so the curve reflects the model's overall
    confidence calibration independent of prompting strategy.  Buckets with
    fewer than 3 samples are dropped to avoid noisy single-point estimates.

    Returns {model_name: DataFrame(mean_conf, accuracy, count)}.
    """
    if "model_name" not in df.columns:
        df = df.copy()
        df["model_name"] = "unknown"

    df = df.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence", "correct"])

    result: dict[str, pd.DataFrame] = {}
    for model, group in df.groupby("model_name"):
        group = group.copy()
        group["_bin"] = pd.cut(
            group["confidence"], bins=n_bins, labels=False, include_lowest=True
        )
        records = []
        for _, bucket in group.groupby("_bin", observed=True):
            if len(bucket) < 3:
                continue
            records.append({
                "mean_conf": bucket["confidence"].mean() / 100,
                "accuracy": float(bucket["correct"].mean()),
                "count": int(len(bucket)),
            })
        result[model] = (
            pd.DataFrame(records) if records
            else pd.DataFrame(columns=["mean_conf", "accuracy", "count"])
        )
    return result


# ── single-model chart (original style) ──────────────────────────────────────

def plot_bar(ax, conditions, values, title, ylabel, color="steelblue", bar_width=0.8, y_max=1.0):
    x = range(len(conditions))
    colors = color if isinstance(color, (list, tuple)) else [color] * len(conditions)
    ax.bar(x, values, width=bar_width, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.grid(False)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=12)


def hallucination_colors(conditions):
    return ["red" if c == "confident" else "orange" for c in conditions]


def accuracy_colors(conditions):
    return ["green" if c == "baseline" else "steelblue" for c in conditions]


def abstain_colors(conditions):
    return ["purple" if c == "chain_of_thought" else "seagreen" for c in conditions]


# ── multi-model chart ─────────────────────────────────────────────────────────

def plot_grouped_bars(ax, conditions, models, values_by_model, title, ylabel, y_max=1.0):
    """One bar cluster per condition, one bar per model inside each cluster."""
    n, m = len(conditions), len(models)
    bar_w = 0.8 / m
    x = np.arange(n)

    for i, model in enumerate(models):
        offset = (i - (m - 1) / 2) * bar_w
        vals = values_by_model[model]
        ax.bar(
            x + offset, vals,
            width=bar_w * 0.9,
            label=model,
            color=_MODEL_COLORS[i % len(_MODEL_COLORS)],
            edgecolor="black",
            linewidth=0.5,
        )
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(False)


# ── calibration chart ─────────────────────────────────────────────────────────

def plot_calibration_curves(ax, cal_data: dict, models: list) -> None:
    """
    Reliability diagram: mean predicted confidence vs actual accuracy per bucket.

    The dashed diagonal represents perfect calibration.
    Points above it → model is underconfident.
    Points below it → model is overconfident.
    One curve per model when multiple models are present.
    """
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="gray", linewidth=1,
        label="Perfect calibration", zorder=0,
    )

    for i, model in enumerate(models):
        data = cal_data.get(model)
        if data is None or data.empty:
            continue
        ax.plot(
            data["mean_conf"], data["accuracy"],
            marker="o", linestyle="-",
            color=_MODEL_COLORS[i % len(_MODEL_COLORS)],
            label=model, linewidth=1.5, markersize=5,
        )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Calibration Curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle=":")


# ── helpers ───────────────────────────────────────────────────────────────────

def _values_by_model(summary: pd.DataFrame, conditions: list, models: list, col: str) -> dict:
    """Return {model: [value_per_condition]} aligned to the given conditions list."""
    lookup = summary.set_index(["model_name", "condition"])[col]
    return {m: [float(lookup.get((m, c), 0.0)) for c in conditions] for m in models}


def _auto_y_max(values_by_model: dict, headroom: float = 0.15) -> float:
    peak = max(v for vals in values_by_model.values() for v in vals)
    return min(1.0, round(peak + headroom, 1))


# ── plot paths ────────────────────────────────────────────────────────────────

def _plot_single_model(
    summary: pd.DataFrame, conditions: list, cal_data: dict, plots_dir: Path
) -> None:
    # Align rows to the caller-supplied condition order.
    s = summary.set_index("condition").reindex(conditions).reset_index()

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plot_bar(
        ax1, conditions, s["accuracy"].tolist(),
        "Baseline Delivers Highest Accuracy", "Accuracy",
        accuracy_colors(conditions), bar_width=0.95,
    )
    plt.tight_layout()
    fig1.savefig(plots_dir / "accuracy_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_bar(
        ax2, conditions, s["hallucination_rate"].tolist(),
        "Forcing Confidence Doubles Hallucination Rate", "Hallucination Rate",
        hallucination_colors(conditions), bar_width=0.95, y_max=0.4,
    )
    plt.tight_layout()
    fig2.savefig(plots_dir / "hallucination_rate_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    plot_bar(
        ax3, conditions, s["abstain_rate"].tolist(),
        "Chain-of-Thought Raises Abstention", "Abstain Rate",
        abstain_colors(conditions), bar_width=0.95, y_max=0.4,
    )
    plt.tight_layout()
    fig3.savefig(plots_dir / "abstain_rate_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    plot_calibration_curves(ax4, cal_data, list(cal_data.keys()))
    plt.tight_layout()
    fig4.savefig(plots_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)


def _plot_multi_model(
    summary: pd.DataFrame, conditions: list, models: list, cal_data: dict, plots_dir: Path
) -> None:
    acc_vals = _values_by_model(summary, conditions, models, "accuracy")
    hall_vals = _values_by_model(summary, conditions, models, "hallucination_rate")
    abst_vals = _values_by_model(summary, conditions, models, "abstain_rate")

    fig_w = max(7, len(conditions) * 1.5)

    fig1, ax1 = plt.subplots(figsize=(fig_w, 5))
    plot_grouped_bars(ax1, conditions, models, acc_vals, "Accuracy by Condition", "Accuracy")
    plt.tight_layout()
    fig1.savefig(plots_dir / "accuracy_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(fig_w, 5))
    plot_grouped_bars(
        ax2, conditions, models, hall_vals,
        "Hallucination Rate by Condition", "Hallucination Rate",
        y_max=_auto_y_max(hall_vals),
    )
    plt.tight_layout()
    fig2.savefig(plots_dir / "hallucination_rate_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(fig_w, 5))
    plot_grouped_bars(
        ax3, conditions, models, abst_vals,
        "Abstain Rate by Condition", "Abstain Rate",
        y_max=_auto_y_max(abst_vals),
    )
    plt.tight_layout()
    fig3.savefig(plots_dir / "abstain_rate_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    plot_calibration_curves(ax4, cal_data, models)
    plt.tight_layout()
    fig4.savefig(plots_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=project_root() / "results" / "scored.csv",
        type=Path,
        help="Input scored CSV",
    )
    parser.add_argument(
        "--summary",
        default=project_root() / "results" / "summary.csv",
        type=Path,
        help="Output summary CSV",
    )
    parser.add_argument(
        "--plots-dir",
        default=project_root() / "results" / "plots",
        type=Path,
        help="Directory for plot PNGs",
    )
    args = parser.parse_args()

    df = load_scored(args.input)
    summary = compute_summary(df)
    cal_data = compute_calibration_data(df)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"Summary saved: {args.summary}")

    # Print per-model Brier scores aggregated across all conditions.
    model_brier = (
        summary.groupby("model_name")["brier_score"]
        .mean()
        .dropna()
        .sort_values()
    )
    if not model_brier.empty:
        print("\nBrier scores — lower is better calibrated (0 = perfect, 1 = worst):")
        for model, score in model_brier.items():
            print(f"  {model}: {score:.4f}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    models = summary["model_name"].unique().tolist()
    # Preserve the order conditions appear in the data (first-seen across all models).
    conditions = list(dict.fromkeys(summary["condition"]))

    if len(models) > 1:
        _plot_multi_model(summary, conditions, models, cal_data, args.plots_dir)
    else:
        _plot_single_model(
            summary[summary["model_name"] == models[0]].reset_index(drop=True),
            conditions,
            cal_data,
            args.plots_dir,
        )

    print(f"Plots saved to {args.plots_dir}/")


if __name__ == "__main__":
    main()
