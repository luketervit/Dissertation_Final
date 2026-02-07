"""
Batch Validation of 100 Thread Simulations

Classifies all tweets (real + simulated) across 100 threads through 5 RoBERTa
models, computes per-thread metrics, and generates 6 dissertation-quality figures.

Usage:
    python scripts/validate_batch_100.py [--device auto|cuda|cpu] [--resume]

Output directory: batch_analysis_100/
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon, pearsonr
from transformers import pipeline
import torch
from tqdm import tqdm

# ── Reuse helpers from validate_comprehensive.py ─────────────────────────────

# Add project root to path so we can import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_comprehensive import (
    POLITICAL_LABEL_MAP,
    load_models,
    load_thread,
    compute_distribution,
    compute_jsd,
)

# ── Constants ────────────────────────────────────────────────────────────────

BATCH_DIR = PROJECT_ROOT / "batch_output_100"
OUTPUT_DIR = PROJECT_ROOT / "batch_analysis_100"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_PATH = OUTPUT_DIR / "all_tweets_classified.csv"
PER_THREAD_PATH = OUTPUT_DIR / "per_thread_results.csv"
AGGREGATE_PATH = OUTPUT_DIR / "aggregate_results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
POLITICAL_LABELS = ["Left", "Center", "Right"]


# ── Classification with full sentiment probabilities ─────────────────────────


def classify_tweet_full(text: str, models: dict) -> dict:
    """Classify a tweet through all 5 models.

    Returns the same fields as validate_comprehensive.classify_tweet plus
    sentiment_pos / sentiment_neg / sentiment_neu raw probabilities so we
    can compute a continuous sentiment score.
    """
    text = str(text)[:512]

    p = models["political"](text)[0]
    e = models["emotion"](text)[0]
    h = models["hate"](text)[0]
    o = models["offensive"](text)[0]

    # Sentiment: get ALL class probabilities
    sn_all = models["sentiment"](text, top_k=None)
    sn_probs = {r["label"]: r["score"] for r in sn_all}
    sn_top = max(sn_all, key=lambda x: x["score"])

    p_pos = sn_probs.get("positive", 0.0)
    p_neg = sn_probs.get("negative", 0.0)
    p_neu = sn_probs.get("neutral", 0.0)

    return {
        "political_label": POLITICAL_LABEL_MAP.get(p["label"], p["label"]),
        "political_score": p["score"],
        "emotion_label": e["label"],
        "emotion_score": e["score"],
        "sentiment_label": sn_top["label"],
        "sentiment_score": sn_top["score"],
        "sentiment_pos": p_pos,
        "sentiment_neg": p_neg,
        "sentiment_neu": p_neu,
        "sentiment_continuous": p_pos - p_neg,  # [-1, 1]
        "hate_score": h["score"] if h["label"] == "HATE" else 1 - h["score"],
        "offensive_score": (
            o["score"] if o["label"] == "OFFENSIVE" else 1 - o["score"]
        ),
    }


# ── Step 1: Classify all tweets ─────────────────────────────────────────────


def classify_all_threads(models: dict, resume: bool = True) -> pd.DataFrame:
    """Classify every tweet in all 100 threads. Saves checkpoint after each
    thread so work is never lost."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if it exists
    already_done: set[str] = set()
    rows: list[dict] = []
    if resume and CHECKPOINT_PATH.exists():
        df_existing = pd.read_csv(CHECKPOINT_PATH)
        rows = df_existing.to_dict("records")
        already_done = set(df_existing["thread_id"].unique())
        print(f"Resuming from checkpoint: {len(already_done)} threads already done")

    # Discover thread folders
    thread_dirs = sorted(
        [d for d in BATCH_DIR.iterdir() if d.is_dir() and d.name.startswith("thread_")],
        key=lambda d: d.name,
    )
    print(f"Found {len(thread_dirs)} thread folders")

    for tdir in tqdm(thread_dirs, desc="Threads"):
        thread_id = tdir.name
        if thread_id in already_done:
            continue

        real_path = tdir / "thread_metadata.json"
        sim_path = tdir / "simulation_output" / "simulated_thread_metadata.json"

        if not real_path.exists() or not sim_path.exists():
            print(f"  Skipping {thread_id}: missing files")
            continue

        try:
            real_tweets = load_thread(str(real_path))
            sim_tweets = load_thread(str(sim_path))
        except Exception as exc:
            print(f"  Skipping {thread_id}: {exc}")
            continue

        # Classify real tweets
        for tweet in real_tweets:
            dna = classify_tweet_full(tweet["text"], models)
            dna["thread_id"] = thread_id
            dna["source"] = "real"
            dna["text"] = tweet["text"]
            dna["tweet_id"] = tweet.get("tweet_id")
            rows.append(dna)

        # Classify simulated tweets
        for tweet in sim_tweets:
            dna = classify_tweet_full(tweet["text"], models)
            dna["thread_id"] = thread_id
            dna["source"] = "simulated"
            dna["text"] = tweet["text"]
            dna["tweet_id"] = tweet.get("tweet_id")
            rows.append(dna)

        # Save checkpoint after each thread
        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
        already_done.add(thread_id)

    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"Classification complete: {len(df)} tweets across {df['thread_id'].nunique()} threads")
    return df


# ── Step 2: Per-thread metrics ───────────────────────────────────────────────


def compute_per_thread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute validation metrics for each thread."""
    records = []

    for thread_id, tdf in df.groupby("thread_id"):
        real = tdf[tdf["source"] == "real"]
        sim = tdf[tdf["source"] == "simulated"]

        if len(real) == 0 or len(sim) == 0:
            continue

        # Sentiment continuous scores
        real_sent_mean = real["sentiment_continuous"].mean()
        sim_sent_mean = sim["sentiment_continuous"].mean()

        # JSD for categorical dimensions
        real_sent_dist = compute_distribution(real.to_dict("records"), "sentiment_label")
        sim_sent_dist = compute_distribution(sim.to_dict("records"), "sentiment_label")
        sent_jsd = compute_jsd(real_sent_dist, sim_sent_dist)

        real_pol_dist = compute_distribution(real.to_dict("records"), "political_label")
        sim_pol_dist = compute_distribution(sim.to_dict("records"), "political_label")
        pol_jsd = compute_jsd(real_pol_dist, sim_pol_dist)

        real_emo_dist = compute_distribution(real.to_dict("records"), "emotion_label")
        sim_emo_dist = compute_distribution(sim.to_dict("records"), "emotion_label")
        emo_jsd = compute_jsd(real_emo_dist, sim_emo_dist)

        # Aggression
        real_agg = (real["hate_score"] + real["offensive_score"]).mean()
        sim_agg = (sim["hate_score"] + sim["offensive_score"]).mean()

        # Political percentages
        real_right_pct = real_pol_dist.get("Right", 0.0)
        sim_right_pct = sim_pol_dist.get("Right", 0.0)
        real_left_pct = real_pol_dist.get("Left", 0.0)
        sim_left_pct = sim_pol_dist.get("Left", 0.0)

        # Sentiment percentages
        real_neg_pct = real_sent_dist.get("negative", 0.0)
        sim_neg_pct = sim_sent_dist.get("negative", 0.0)

        # Overall weighted accuracy (same formula as validate_comprehensive.py)
        pol_sim_pct = (1 - pol_jsd) * 100
        emo_sim_pct = (1 - emo_jsd) * 100
        sent_sim_pct = (1 - sent_jsd) * 100
        agg_sim_pct = (1 - abs(real_agg - sim_agg)) * 100
        agg_sim_pct = max(agg_sim_pct, 0.0)  # floor at 0

        overall = (
            pol_sim_pct * 0.30
            + emo_sim_pct * 0.20
            + sent_sim_pct * 0.30
            + agg_sim_pct * 0.20
        )

        records.append({
            "thread_id": thread_id,
            "n_real": len(real),
            "n_sim": len(sim),
            "real_sentiment_mean": real_sent_mean,
            "sim_sentiment_mean": sim_sent_mean,
            "sentiment_residual": sim_sent_mean - real_sent_mean,
            "sentiment_jsd": sent_jsd,
            "political_jsd": pol_jsd,
            "emotion_jsd": emo_jsd,
            "real_aggression_mean": real_agg,
            "sim_aggression_mean": sim_agg,
            "real_right_pct": real_right_pct,
            "sim_right_pct": sim_right_pct,
            "real_left_pct": real_left_pct,
            "sim_left_pct": sim_left_pct,
            "real_neg_pct": real_neg_pct,
            "sim_neg_pct": sim_neg_pct,
            "overall_accuracy": overall,
        })

    result = pd.DataFrame(records)
    result.to_csv(PER_THREAD_PATH, index=False)
    print(f"Per-thread metrics saved: {PER_THREAD_PATH} ({len(result)} threads)")
    return result


# ── Step 3: Figures ──────────────────────────────────────────────────────────


def make_figures(per_thread: pd.DataFrame, all_tweets: pd.DataFrame) -> None:
    """Generate 6 dissertation-quality figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
    })

    _fig1_sentiment_residual(per_thread)
    _fig2_jsd_boxplot(per_thread)
    _fig3_aggression_scatter(per_thread)
    _fig4_overall_accuracy_histogram(per_thread)
    _fig5_political_aggregate(all_tweets)
    _fig6_sentiment_overlay(all_tweets)

    print(f"All 6 figures saved to {FIGURES_DIR}/")


def _fig1_sentiment_residual(pt: pd.DataFrame) -> None:
    """Histogram of per-thread sentiment residuals (sim - real)."""
    residuals = pt["sentiment_residual"].values
    mu, sigma = residuals.mean(), residuals.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=25, edgecolor="black", alpha=0.75, color="#4CAF50")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect match")
    ax.axvline(mu, color="navy", linestyle="-", linewidth=1.5,
               label=f"Mean = {mu:.3f}")
    ax.set_xlabel("Sentiment Residual (simulated − real)")
    ax.set_ylabel("Number of Threads")
    ax.set_title("Distribution of Sentiment Residuals Across 100 Threads")
    ax.legend()
    ax.text(0.97, 0.95, f"μ = {mu:.3f}\nσ = {sigma:.3f}\nn = {len(residuals)}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_sentiment_residual_histogram.png", dpi=300)
    plt.close(fig)
    print("  Fig 1: sentiment residual histogram")


def _fig2_jsd_boxplot(pt: pd.DataFrame) -> None:
    """Boxplot of JSD values across 3 dimensions."""
    data = [
        pt["sentiment_jsd"].values,
        pt["political_jsd"].values,
        pt["emotion_jsd"].values,
    ]
    labels = ["Sentiment JSD", "Political JSD", "Emotion JSD"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colours = ["#2196F3", "#FF9800", "#9C27B0"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=1,
               label='JSD = 0.15 ("good" threshold)')
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("Validation Quality Across 100 Threads")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate medians
    for i, d in enumerate(data, 1):
        med = np.median(d)
        ax.text(i, med + 0.005, f"{med:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_jsd_boxplot.png", dpi=300)
    plt.close(fig)
    print("  Fig 2: JSD boxplot")


def _fig3_aggression_scatter(pt: pd.DataFrame) -> None:
    """Scatter: real vs simulated mean aggression per thread."""
    x = pt["real_aggression_mean"].values
    y = pt["sim_aggression_mean"].values
    r, p_val = pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.6, s=40, edgecolors="black", linewidths=0.5,
               color="#E91E63")
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x (perfect match)")
    ax.set_xlabel("Real Thread Mean Aggression")
    ax.set_ylabel("Simulated Thread Mean Aggression")
    ax.set_title("Aggression: Real vs Simulated (100 Threads)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.05, f"Pearson r = {r:.3f}\np = {p_val:.2e}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_aggression_scatter.png", dpi=300)
    plt.close(fig)
    print("  Fig 3: aggression scatter")


def _fig4_overall_accuracy_histogram(pt: pd.DataFrame) -> None:
    """Histogram of overall weighted accuracy."""
    acc = pt["overall_accuracy"].values
    mu, sigma = acc.mean(), acc.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(acc, bins=20, edgecolor="black", alpha=0.75, color="#FF9800")

    # Threshold lines
    for val, label, colour in [
        (80, "EXCELLENT (≥80%)", "#4CAF50"),
        (60, "GOOD (≥60%)", "#2196F3"),
        (40, "FAIR (≥40%)", "#FFC107"),
    ]:
        ax.axvline(val, color=colour, linestyle="--", linewidth=1.2, label=label)

    ax.set_xlabel("Overall Weighted Accuracy (%)")
    ax.set_ylabel("Number of Threads")
    ax.set_title("Distribution of Overall Accuracy Across 100 Threads")
    ax.legend(loc="upper left", fontsize=9)

    n_exc = (acc >= 80).sum()
    n_good = ((acc >= 60) & (acc < 80)).sum()
    n_fair = ((acc >= 40) & (acc < 60)).sum()
    n_poor = (acc < 40).sum()
    ax.text(0.97, 0.95,
            f"μ = {mu:.1f}%  σ = {sigma:.1f}%\n"
            f"EXCELLENT: {n_exc}\nGOOD: {n_good}\nFAIR: {n_fair}\nPOOR: {n_poor}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_overall_accuracy_histogram.png", dpi=300)
    plt.close(fig)
    print("  Fig 4: overall accuracy histogram")


def _fig5_political_aggregate(df: pd.DataFrame) -> None:
    """Side-by-side bar chart of aggregate political distributions."""
    real = df[df["source"] == "real"]
    sim = df[df["source"] == "simulated"]

    real_dist = compute_distribution(real.to_dict("records"), "political_label")
    sim_dist = compute_distribution(sim.to_dict("records"), "political_label")

    labels = POLITICAL_LABELS
    x = np.arange(len(labels))
    width = 0.35
    real_vals = [real_dist.get(l, 0) for l in labels]
    sim_vals = [sim_dist.get(l, 0) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, real_vals, width, label="Real", color="#2196F3",
                   alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, sim_vals, width, label="Simulated", color="#FF9800",
                   alpha=0.85, edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Proportion")
    ax.set_title("Aggregate Political Distribution: Real vs Simulated (All 100 Threads)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(real_vals), max(sim_vals)) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_political_aggregate.png", dpi=300)
    plt.close(fig)
    print("  Fig 5: political aggregate")


def _fig6_sentiment_overlay(df: pd.DataFrame) -> None:
    """Overlapping density histograms of continuous sentiment score."""
    real_scores = df.loc[df["source"] == "real", "sentiment_continuous"].values
    sim_scores = df.loc[df["source"] == "simulated", "sentiment_continuous"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 50)
    ax.hist(real_scores, bins=bins, density=True, alpha=0.55, label="Real",
            color="#2196F3", edgecolor="black", linewidth=0.3)
    ax.hist(sim_scores, bins=bins, density=True, alpha=0.55, label="Simulated",
            color="#FF9800", edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Sentiment Score (P(pos) − P(neg))")
    ax.set_ylabel("Density")
    ax.set_title("Sentiment Distribution: Real vs Simulated (All Tweets Pooled)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_sentiment_distribution_overlay.png", dpi=300)
    plt.close(fig)
    print("  Fig 6: sentiment overlay")


# ── Step 4: Summary statistics ───────────────────────────────────────────────


def write_summary(per_thread: pd.DataFrame, all_tweets: pd.DataFrame) -> None:
    """Write human-readable summary + aggregate JSON."""
    n = len(per_thread)
    acc = per_thread["overall_accuracy"].values

    # Wilcoxon signed-rank test on per-thread mean sentiment
    real_means = per_thread["real_sentiment_mean"].values
    sim_means = per_thread["sim_sentiment_mean"].values
    diffs = sim_means - real_means
    if np.any(diffs != 0):
        stat, p_val = wilcoxon(real_means, sim_means)
    else:
        stat, p_val = 0.0, 1.0

    n_exc = int((acc >= 80).sum())
    n_good = int(((acc >= 60) & (acc < 80)).sum())
    n_fair = int(((acc >= 40) & (acc < 60)).sum())
    n_poor = int((acc < 40).sum())

    lines = [
        "=" * 72,
        "  BATCH VALIDATION SUMMARY — 100 THREADS",
        "=" * 72,
        "",
        f"  Threads analysed:  {n}",
        f"  Total tweets classified:  {len(all_tweets)}",
        f"    Real:      {(all_tweets['source'] == 'real').sum()}",
        f"    Simulated: {(all_tweets['source'] == 'simulated').sum()}",
        "",
        "  Per-Thread Metrics (mean ± std):",
        f"    Sentiment JSD:      {per_thread['sentiment_jsd'].mean():.4f} ± {per_thread['sentiment_jsd'].std():.4f}",
        f"    Political JSD:      {per_thread['political_jsd'].mean():.4f} ± {per_thread['political_jsd'].std():.4f}",
        f"    Emotion JSD:        {per_thread['emotion_jsd'].mean():.4f} ± {per_thread['emotion_jsd'].std():.4f}",
        f"    Sentiment residual: {per_thread['sentiment_residual'].mean():.4f} ± {per_thread['sentiment_residual'].std():.4f}",
        f"    Real aggression:    {per_thread['real_aggression_mean'].mean():.4f} ± {per_thread['real_aggression_mean'].std():.4f}",
        f"    Sim aggression:     {per_thread['sim_aggression_mean'].mean():.4f} ± {per_thread['sim_aggression_mean'].std():.4f}",
        f"    Overall accuracy:   {acc.mean():.1f}% ± {acc.std():.1f}%",
        "",
        "  Rating Breakdown:",
        f"    EXCELLENT (≥80%): {n_exc}",
        f"    GOOD (60–80%):    {n_good}",
        f"    FAIR (40–60%):    {n_fair}",
        f"    POOR (<40%):      {n_poor}",
        "",
        "  Wilcoxon Signed-Rank Test (per-thread mean sentiment, real vs sim):",
        f"    Statistic: {stat:.2f}",
        f"    p-value:   {p_val:.4e}",
        f"    Significant (α=0.05): {'Yes' if p_val < 0.05 else 'No'}",
        "",
        "=" * 72,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)

    SUMMARY_PATH.write_text(summary_text + "\n")
    print(f"\nSaved: {SUMMARY_PATH}")

    # Aggregate JSON
    r_agg, _ = pearsonr(
        per_thread["real_aggression_mean"].values,
        per_thread["sim_aggression_mean"].values,
    )
    aggregate = {
        "n_threads": n,
        "n_tweets_total": len(all_tweets),
        "n_tweets_real": int((all_tweets["source"] == "real").sum()),
        "n_tweets_simulated": int((all_tweets["source"] == "simulated").sum()),
        "sentiment_jsd_mean": float(per_thread["sentiment_jsd"].mean()),
        "sentiment_jsd_std": float(per_thread["sentiment_jsd"].std()),
        "political_jsd_mean": float(per_thread["political_jsd"].mean()),
        "political_jsd_std": float(per_thread["political_jsd"].std()),
        "emotion_jsd_mean": float(per_thread["emotion_jsd"].mean()),
        "emotion_jsd_std": float(per_thread["emotion_jsd"].std()),
        "sentiment_residual_mean": float(per_thread["sentiment_residual"].mean()),
        "sentiment_residual_std": float(per_thread["sentiment_residual"].std()),
        "aggression_pearson_r": float(r_agg),
        "overall_accuracy_mean": float(acc.mean()),
        "overall_accuracy_std": float(acc.std()),
        "rating_excellent": n_exc,
        "rating_good": n_good,
        "rating_fair": n_fair,
        "rating_poor": n_poor,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_val),
    }
    with open(AGGREGATE_PATH, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved: {AGGREGATE_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Batch validate 100 thread simulations")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint (default: true)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore checkpoint")
    args = parser.parse_args()

    resume = not args.no_resume

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Classify all tweets
    models = load_models(args.device)
    df = classify_all_threads(models, resume=resume)

    # Step 2: Per-thread metrics
    per_thread = compute_per_thread_metrics(df)

    # Step 3: Figures
    print("\nGenerating figures...")
    make_figures(per_thread, df)

    # Step 4: Summary
    print()
    write_summary(per_thread, df)

    print("\nDone! All outputs in batch_analysis_100/")


if __name__ == "__main__":
    main()
