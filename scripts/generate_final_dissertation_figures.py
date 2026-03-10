#!/usr/bin/env python3
"""Build the curated dissertation figure pack in one folder.

The pack intentionally excludes figures that were judged weak or redundant:
- no combined pipeline overview
- no standalone fidelity-distribution histogram
- no fidelity sub-metric boxplot
- no cross-model comparison figure
- no main-text case-study figure

The output is written to dissertation_figures/final.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "dissertation_figures" / "final"


def ensure_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_old_pngs() -> None:
    for path in OUT_DIR.glob("*.png"):
        path.unlink()


def copy(src: str, dst: str) -> None:
    shutil.copy2(PROJECT_ROOT / src, OUT_DIR / dst)


def build_static_pack() -> None:
    copy(
        "dissertation_figures/implementation/personality_extraction_pipeline.png",
        "01_personality_extraction_pipeline.png",
    )
    copy(
        "dissertation_figures/implementation/simulation_pipeline.png",
        "02_simulation_pipeline.png",
    )
    copy(
        "dissertation_figures/fidelity_scores/02_fidelity_scores_ranked.png",
        "09_fidelity_scores_ranked.png",
    )
    copy(
        "dissertation_figures/parameter_sweep/02_metric_heatmap.png",
        "10_ablation_metric_heatmap.png",
    )
    copy(
        "dissertation_figures/parameter_sweep/01_composite_leaderboard.png",
        "11_ablation_leaderboard.png",
    )


def _load_batch_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per_thread = pd.read_csv(PROJECT_ROOT / "analysis" / "batch_analysis_new" / "per_thread_results.csv")
    all_tweets = pd.read_csv(PROJECT_ROOT / "analysis" / "batch_analysis_new" / "all_tweets_classified.csv")
    aggregate = pd.read_json(
        PROJECT_ROOT / "analysis" / "batch_analysis_new" / "aggregate_results.json",
        typ="series",
    ).to_dict()
    return per_thread, all_tweets, aggregate


def generate_jsd_boxplot(per_thread: pd.DataFrame) -> None:
    metrics = [
        ("sentiment_jsd", "Sentiment JSD", "#5DA5DA"),
        ("political_jsd", "Political JSD", "#F0B44C"),
        ("emotion_jsd", "Emotion JSD", "#B276C2"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=220)
    fig.patch.set_facecolor("white")

    data = [per_thread[col].dropna().values for col, _, _ in metrics]
    box = ax.boxplot(
        data,
        tick_labels=[label for _, label, _ in metrics],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#E67E22", "linewidth": 1.6},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, (_, _, color) in zip(box["boxes"], metrics):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=1.3, alpha=0.8)
    for i, (col, _, _) in enumerate(metrics, start=1):
        mean = per_thread[col].mean()
        ax.text(i, mean + 0.012, f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Validation quality across 100 threads", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_jsd_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def generate_sentiment_residual_histogram(per_thread: pd.DataFrame, aggregate: dict) -> None:
    residuals = per_thread["sentiment_residual"].dropna().values
    mean = aggregate["sentiment_residual_mean"]
    std = aggregate["sentiment_residual_std"]

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.hist(residuals, bins=24, color="#6EB56E", edgecolor="#333333", alpha=0.92)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.6, label="Perfect match")
    ax.axvline(mean, color="#1F3A93", linewidth=1.6, label=f"Mean = {mean:.3f}")
    ax.text(
        0.95,
        0.95,
        f"\u03bc = {mean:.3f}\n\u03c3 = {std:.3f}\nn = {len(residuals)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#F2E2BA", "alpha": 0.95},
    )
    ax.set_xlabel("Sentiment residual (simulated \u2212 real)")
    ax.set_ylabel("Threads")
    ax.set_title("Distribution of sentiment residuals across 100 threads", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_sentiment_residual_histogram.png", bbox_inches="tight")
    plt.close(fig)


def generate_aggression_scatter(per_thread: pd.DataFrame, aggregate: dict) -> None:
    x = per_thread["real_aggression_mean"].values
    y = per_thread["sim_aggression_mean"].values
    upper = max(x.max(), y.max()) * 1.08

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=220)
    fig.patch.set_facecolor("white")
    ax.scatter(x, y, s=28, color="#E85D8D", edgecolors="#444444", linewidths=0.4, alpha=0.88)
    ax.plot([0, upper], [0, upper], linestyle="--", color="#222222", linewidth=1.0, label="y = x (perfect match)")
    ax.text(
        0.97,
        0.06,
        f"Pearson r = {aggregate['aggression_pearson_r']:.3f}\np = {aggregate['aggression_pearson_p']:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round", "facecolor": "#F2E2BA", "alpha": 0.95},
    )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Real thread mean aggression")
    ax.set_ylabel("Simulated thread mean aggression")
    ax.set_title("Aggression: real vs simulated", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_aggression_scatter.png", bbox_inches="tight")
    plt.close(fig)


def generate_overall_accuracy_histogram(per_thread: pd.DataFrame, aggregate: dict) -> None:
    scores = per_thread["overall_accuracy"].dropna().values

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.hist(scores, bins=22, color="#F5B041", edgecolor="#333333", alpha=0.96)
    thresholds = [
        (80, "#4CAF50", "EXCELLENT (\u226580%)"),
        (60, "#1E90FF", "GOOD (\u226560%)"),
        (40, "#F4B400", "FAIR (\u226540%)"),
    ]
    for value, color, label in thresholds:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.3, alpha=0.9, label=label)
    ax.text(
        0.97,
        0.95,
        (
            f"\u03bc = {aggregate['overall_accuracy_mean']:.1f}%  "
            f"\u03c3 = {aggregate['overall_accuracy_std']:.1f}%\n"
            f"EXCELLENT: {aggregate['rating_excellent']}\n"
            f"GOOD: {aggregate['rating_good']}\n"
            f"FAIR: {aggregate['rating_fair']}\n"
            f"POOR: {aggregate['rating_poor']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "#F2E2BA", "alpha": 0.95},
    )
    ax.set_xlabel("Overall weighted accuracy (%)")
    ax.set_ylabel("Threads")
    ax.set_title("Distribution of overall accuracy", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_overall_accuracy_histogram.png", bbox_inches="tight")
    plt.close(fig)


def generate_political_aggregate(all_tweets: pd.DataFrame) -> None:
    order = ["Left", "Center", "Right"]
    grouped = (
        all_tweets.groupby(["source", "political_label"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
    )
    proportions = grouped.div(grouped.sum(axis=1), axis=0)
    x = np.arange(len(order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.3, 5.2), dpi=220)
    fig.patch.set_facecolor("white")
    real = proportions.loc["real", order].values
    sim = proportions.loc["simulated", order].values
    bars_real = ax.bar(x - width / 2, real, width=width, color="#4C9BE8", label="Real")
    bars_sim = ax.bar(x + width / 2, sim, width=width, color="#F5A623", label="Simulated")
    for bars in (bars_real, bars_sim):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h*100:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Proportion")
    ax.set_title("Aggregate political distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_political_aggregate.png", bbox_inches="tight")
    plt.close(fig)


def generate_sentiment_distribution_comparison() -> None:
    df = pd.read_csv(PROJECT_ROOT / "analysis" / "batch_analysis_new" / "all_tweets_classified.csv")
    real_scores = df.loc[df["source"] == "real", "sentiment_continuous"].dropna().values
    sim_scores = df.loc[df["source"] == "simulated", "sentiment_continuous"].dropna().values
    bins = np.linspace(-1, 1, 45)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)
    fig.patch.set_facecolor("white")

    ax.hist(
        real_scores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="#2C7FB8",
        label="Real",
    )
    ax.hist(
        sim_scores,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        color="#D95F0E",
        label="Simulated",
    )

    ax.axvline(real_scores.mean(), color="#2C7FB8", linestyle=":", linewidth=1.6, alpha=0.8)
    ax.axvline(sim_scores.mean(), color="#D95F0E", linestyle=":", linewidth=1.6, alpha=0.8)
    ax.set_xlabel("Sentiment score  ($P(\\mathrm{pos}) - P(\\mathrm{neg})$)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pooled sentiment score distribution across all 100 threads", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_sentiment_distribution_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dir()
    clear_old_pngs()
    build_static_pack()
    per_thread, all_tweets, aggregate = _load_batch_results()
    generate_jsd_boxplot(per_thread)
    generate_sentiment_residual_histogram(per_thread, aggregate)
    generate_aggression_scatter(per_thread, aggregate)
    generate_overall_accuracy_histogram(per_thread, aggregate)
    generate_political_aggregate(all_tweets)
    generate_sentiment_distribution_comparison()
    print(f"Final dissertation figure pack written to {OUT_DIR}")


if __name__ == "__main__":
    main()
