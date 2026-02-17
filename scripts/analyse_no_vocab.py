"""
Full validation analysis for the no_vocab ablation config.

Reuses already-classified tweets from the sweep analysis checkpoint.
Generates 8 dissertation-quality figures + summary report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_comprehensive import compute_distribution, compute_jsd

# ── Paths ────────────────────────────────────────────────────────────────────

SWEEP_DIR = PROJECT_ROOT / "parameter_sweep"
REAL_DIR = PROJECT_ROOT / "batch_simulations_reconstructed"
CHECKPOINT = SWEEP_DIR / "analysis" / "all_tweets_classified.csv"
OUTPUT_DIR = PROJECT_ROOT / "batch_analysis_no_vocab"
FIGURES_DIR = OUTPUT_DIR / "figures"

VARIANT_LABEL = "0-Shot No Vocab"
POLITICAL_LABELS = ["Left", "Center", "Right"]


# ── Load data ────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Load no_vocab + real tweets from sweep checkpoint."""
    df = pd.read_csv(CHECKPOINT)
    # Keep real tweets and no_vocab simulated tweets
    mask = (df["config"] == "REAL") | (df["config"] == "no_vocab")
    subset = df[mask].copy()
    # Normalise source column
    subset.loc[subset["config"] == "REAL", "source"] = "real"
    subset.loc[subset["config"] == "no_vocab", "source"] = "simulated"
    # Use thread column as thread_id
    subset["thread_id"] = subset["thread"]
    print(f"Loaded {len(subset)} tweets: "
          f"{(subset.source=='real').sum()} real, "
          f"{(subset.source=='simulated').sum()} simulated, "
          f"{subset.thread_id.nunique()} threads")
    return subset


# ── Per-thread metrics ───────────────────────────────────────────────────────


def compute_per_thread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for thread_id, tdf in df.groupby("thread_id"):
        real = tdf[tdf["source"] == "real"]
        sim = tdf[tdf["source"] == "simulated"]
        if len(real) == 0 or len(sim) == 0:
            continue

        # Sentiment
        real_sent_mean = real["sentiment_continuous"].mean()
        sim_sent_mean = sim["sentiment_continuous"].mean()
        real_sent_dist = compute_distribution(real.to_dict("records"), "sentiment_label")
        sim_sent_dist = compute_distribution(sim.to_dict("records"), "sentiment_label")
        sent_jsd = compute_jsd(real_sent_dist, sim_sent_dist)

        # Political
        real_pol_dist = compute_distribution(real.to_dict("records"), "political_label")
        sim_pol_dist = compute_distribution(sim.to_dict("records"), "political_label")
        pol_jsd = compute_jsd(real_pol_dist, sim_pol_dist)

        # Emotion
        real_emo_dist = compute_distribution(real.to_dict("records"), "emotion_label")
        sim_emo_dist = compute_distribution(sim.to_dict("records"), "emotion_label")
        emo_jsd = compute_jsd(real_emo_dist, sim_emo_dist)

        # Aggression
        real_agg = (real["hate_score"] + real["offensive_score"]).mean()
        sim_agg = (sim["hate_score"] + sim["offensive_score"]).mean()

        # Percentages
        real_right_pct = real_pol_dist.get("Right", 0.0)
        sim_right_pct = sim_pol_dist.get("Right", 0.0)
        real_neg_pct = real_sent_dist.get("negative", 0.0)
        sim_neg_pct = sim_sent_dist.get("negative", 0.0)

        # Overall accuracy
        overall = (
            (1 - pol_jsd) * 0.30
            + (1 - emo_jsd) * 0.20
            + (1 - sent_jsd) * 0.30
            + max((1 - abs(real_agg - sim_agg)), 0.0) * 0.20
        ) * 100

        # Thread structure (simulated)
        sim_hist_path = (
            SWEEP_DIR / "no_vocab" / thread_id
            / "simulation_output" / "thread_history.json"
        )
        if sim_hist_path.exists():
            with open(sim_hist_path) as f:
                sim_history = json.load(f)
            depths = [p.get("depth", 0) for p in sim_history]
            sim_max_depth = max(depths)
            sim_mean_depth = float(np.mean(depths))
            sim_subthreads = sum(1 for p in sim_history if p.get("depth") == 1)
            sim_total = len(sim_history)
        else:
            sim_max_depth = 0
            sim_mean_depth = 0.0
            sim_subthreads = 0
            sim_total = len(sim)

        # Real structure (flat)
        real_meta_path = REAL_DIR / thread_id / "thread_metadata.json"
        if real_meta_path.exists():
            with open(real_meta_path) as f:
                meta = json.load(f)
            events = meta.get("temporal_events", [])
            root_count = sum(1 for e in events if e.get("is_root"))
            real_total = len(events)
            real_subthreads = real_total - root_count
        else:
            real_total = len(real)
            real_subthreads = len(real)

        records.append({
            "thread_id": thread_id,
            "n_real": len(real), "n_sim": len(sim),
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
            "real_neg_pct": real_neg_pct,
            "sim_neg_pct": sim_neg_pct,
            "overall_accuracy": overall,
            "real_total_posts": real_total,
            "sim_total_posts": sim_total,
            "real_max_depth": 1,
            "sim_max_depth": sim_max_depth,
            "real_mean_depth": (real_total - 1) / real_total if real_total > 0 else 0,
            "sim_mean_depth": sim_mean_depth,
            "real_num_subthreads": real_subthreads,
            "sim_num_subthreads": sim_subthreads,
        })

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_DIR / "per_thread_results.csv", index=False)
    print(f"Per-thread metrics: {len(result)} threads")
    return result


# ── Figures ──────────────────────────────────────────────────────────────────


def make_figures(pt: pd.DataFrame, df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12, "figure.dpi": 150,
    })

    # Fig 1: Sentiment residual histogram
    residuals = pt["sentiment_residual"].values
    mu, sigma = residuals.mean(), residuals.std()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=15, edgecolor="black", alpha=0.75, color="#4CAF50")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect match")
    ax.axvline(mu, color="navy", linestyle="-", linewidth=1.5, label=f"Mean = {mu:.3f}")
    ax.set_xlabel("Sentiment Residual (simulated − real)")
    ax.set_ylabel("Number of Threads")
    ax.set_title(f"Sentiment Residuals ({VARIANT_LABEL})")
    ax.legend()
    ax.text(0.97, 0.95, f"μ = {mu:.3f}\nσ = {sigma:.3f}\nn = {len(residuals)}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_sentiment_residual_histogram.png", dpi=300)
    plt.close(fig)
    print("  Fig 1: sentiment residual histogram")

    # Fig 2: JSD boxplot
    data = [pt["sentiment_jsd"].values, pt["political_jsd"].values, pt["emotion_jsd"].values]
    labels = ["Sentiment JSD", "Political JSD", "Emotion JSD"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#2196F3", "#FF9800", "#9C27B0"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0.15, color="red", linestyle="--", linewidth=1,
               label='JSD = 0.15 ("good" threshold)')
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title(f"Validation Quality ({VARIANT_LABEL})")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    for i, d in enumerate(data, 1):
        med = np.median(d)
        ax.text(i, med + 0.005, f"{med:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_jsd_boxplot.png", dpi=300)
    plt.close(fig)
    print("  Fig 2: JSD boxplot")

    # Fig 3: Aggression scatter
    x = pt["real_aggression_mean"].values
    y = pt["sim_aggression_mean"].values
    r, p_val = pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.6, s=60, edgecolors="black", linewidths=0.5, color="#E91E63")
    lim = max(x.max(), y.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel("Real Thread Mean Aggression")
    ax.set_ylabel("Simulated Thread Mean Aggression")
    ax.set_title(f"Aggression: Real vs Simulated ({VARIANT_LABEL})")
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

    # Fig 4: Overall accuracy histogram
    acc = pt["overall_accuracy"].values
    mu_a, sigma_a = acc.mean(), acc.std()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(acc, bins=12, edgecolor="black", alpha=0.75, color="#FF9800")
    for val, label, c in [
        (80, "EXCELLENT (≥80%)", "#4CAF50"),
        (60, "GOOD (≥60%)", "#2196F3"),
    ]:
        ax.axvline(val, color=c, linestyle="--", linewidth=1.2, label=label)
    ax.set_xlabel("Overall Weighted Accuracy (%)")
    ax.set_ylabel("Number of Threads")
    ax.set_title(f"Overall Accuracy ({VARIANT_LABEL})")
    ax.legend(loc="upper left", fontsize=9)
    n_exc = (acc >= 80).sum()
    n_good = ((acc >= 60) & (acc < 80)).sum()
    ax.text(0.97, 0.95,
            f"μ = {mu_a:.1f}%  σ = {sigma_a:.1f}%\nEXCELLENT: {n_exc}\nGOOD: {n_good}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_overall_accuracy_histogram.png", dpi=300)
    plt.close(fig)
    print("  Fig 4: overall accuracy histogram")

    # Fig 5: Political aggregate bars
    real = df[df["source"] == "real"]
    sim = df[df["source"] == "simulated"]
    real_dist = compute_distribution(real.to_dict("records"), "political_label")
    sim_dist = compute_distribution(sim.to_dict("records"), "political_label")
    x_pos = np.arange(len(POLITICAL_LABELS))
    width = 0.35
    real_vals = [real_dist.get(l, 0) for l in POLITICAL_LABELS]
    sim_vals = [sim_dist.get(l, 0) for l in POLITICAL_LABELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x_pos - width / 2, real_vals, width, label="Real", color="#2196F3",
                alpha=0.85, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x_pos + width / 2, sim_vals, width, label="Simulated", color="#FF9800",
                alpha=0.85, edgecolor="black", linewidth=0.5)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Proportion")
    ax.set_title(f"Political Distribution ({VARIANT_LABEL})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(POLITICAL_LABELS)
    ax.legend()
    ax.set_ylim(0, max(max(real_vals), max(sim_vals)) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_political_aggregate.png", dpi=300)
    plt.close(fig)
    print("  Fig 5: political aggregate")

    # Fig 6: Sentiment density overlay
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
    ax.set_title(f"Sentiment Distribution ({VARIANT_LABEL})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_sentiment_distribution_overlay.png", dpi=300)
    plt.close(fig)
    print("  Fig 6: sentiment overlay")

    # Fig 7: Depth comparison
    x_d = pt["real_max_depth"].values.astype(float)
    y_d = pt["sim_max_depth"].values.astype(float)
    fig, ax = plt.subplots(figsize=(7, 7))
    x_jitter = x_d + np.random.normal(0, 0.08, len(x_d))
    ax.scatter(x_jitter, y_d, alpha=0.6, s=50, edgecolors="black", linewidths=0.5,
               color="#4CAF50")
    max_val = max(x_d.max(), y_d.max()) + 1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Real Thread Max Depth")
    ax.set_ylabel("Simulated Thread Max Depth")
    ax.set_title(f"Thread Depth ({VARIANT_LABEL})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.05,
            f"Real: all = 1 (flat)\nSim: {y_d.mean():.1f} ± {y_d.std():.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_depth_comparison.png", dpi=300)
    plt.close(fig)
    print("  Fig 7: depth comparison")

    # Fig 8: Subthread comparison
    x_s = pt["real_num_subthreads"].values.astype(float)
    y_s = pt["sim_num_subthreads"].values.astype(float)
    r_s, p_s = pearsonr(x_s, y_s) if len(x_s) > 2 else (0.0, 1.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_s, y_s, alpha=0.6, s=50, edgecolors="black", linewidths=0.5,
               color="#9C27B0")
    max_val = max(x_s.max(), y_s.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Real Thread Subthreads")
    ax.set_ylabel("Simulated Thread Subthreads")
    ax.set_title(f"Subthread Count ({VARIANT_LABEL})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.05,
            f"Pearson r = {r_s:.3f}\np = {p_s:.2e}\n"
            f"Real: {x_s.mean():.1f}\nSim: {y_s.mean():.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_subthread_comparison.png", dpi=300)
    plt.close(fig)
    print("  Fig 8: subthread comparison")


# ── Summary ──────────────────────────────────────────────────────────────────


def write_summary(pt: pd.DataFrame, df: pd.DataFrame) -> None:
    n = len(pt)
    acc = pt["overall_accuracy"].values

    real_means = pt["real_sentiment_mean"].values
    sim_means = pt["sim_sentiment_mean"].values
    diffs = sim_means - real_means
    if np.any(diffs != 0) and len(diffs) >= 5:
        stat, p_val = wilcoxon(real_means, sim_means)
    else:
        stat, p_val = 0.0, 1.0

    r_agg, p_agg = pearsonr(
        pt["real_aggression_mean"].values,
        pt["sim_aggression_mean"].values,
    )

    n_exc = int((acc >= 80).sum())
    n_good = int(((acc >= 60) & (acc < 80)).sum())
    n_fair = int(((acc >= 40) & (acc < 60)).sum())
    n_poor = int((acc < 40).sum())

    lines = [
        "=" * 72,
        f"  BATCH VALIDATION SUMMARY — 10 THREADS ({VARIANT_LABEL.upper()})",
        "=" * 72,
        "",
        f"  Threads analysed:  {n}",
        f"  Total tweets classified:  {len(df)}",
        f"    Real:      {(df['source'] == 'real').sum()}",
        f"    Simulated: {(df['source'] == 'simulated').sum()}",
        "",
        "  Per-Thread Metrics (mean ± std):",
        f"    Sentiment JSD:      {pt['sentiment_jsd'].mean():.4f} ± {pt['sentiment_jsd'].std():.4f}",
        f"    Political JSD:      {pt['political_jsd'].mean():.4f} ± {pt['political_jsd'].std():.4f}",
        f"    Emotion JSD:        {pt['emotion_jsd'].mean():.4f} ± {pt['emotion_jsd'].std():.4f}",
        f"    Sentiment residual: {pt['sentiment_residual'].mean():.4f} ± {pt['sentiment_residual'].std():.4f}",
        f"    Real aggression:    {pt['real_aggression_mean'].mean():.4f} ± {pt['real_aggression_mean'].std():.4f}",
        f"    Sim aggression:     {pt['sim_aggression_mean'].mean():.4f} ± {pt['sim_aggression_mean'].std():.4f}",
        f"    Aggression Pearson r: {r_agg:.3f} (p={p_agg:.2e})",
        f"    Overall accuracy:   {acc.mean():.1f}% ± {acc.std():.1f}%",
        "",
        "  Rating Breakdown:",
        f"    EXCELLENT (≥80%): {n_exc}",
        f"    GOOD (60-80%):    {n_good}",
        f"    FAIR (40-60%):    {n_fair}",
        f"    POOR (<40%):      {n_poor}",
        "",
        "  Wilcoxon Signed-Rank Test (per-thread mean sentiment):",
        f"    Statistic: {stat:.2f}",
        f"    p-value:   {p_val:.4e}",
        f"    Significant (a=0.05): {'Yes' if p_val < 0.05 else 'No'}",
        "",
        "  Thread Structure (Simulated):",
        f"    Max depth (mean ± std):    {pt['sim_max_depth'].mean():.1f} ± {pt['sim_max_depth'].std():.1f}",
        f"    Mean depth (mean ± std):   {pt['sim_mean_depth'].mean():.2f} ± {pt['sim_mean_depth'].std():.2f}",
        f"    Subthreads (mean ± std):   {pt['sim_num_subthreads'].mean():.1f} ± {pt['sim_num_subthreads'].std():.1f}",
        f"    Total posts (mean ± std):  {pt['sim_total_posts'].mean():.1f} ± {pt['sim_total_posts'].std():.1f}",
        "",
        "  Thread Structure (Real):",
        f"    All real threads are flat (max depth = 1)",
        f"    Subthreads (mean ± std):   {pt['real_num_subthreads'].mean():.1f} ± {pt['real_num_subthreads'].std():.1f}",
        f"    Total posts (mean ± std):  {pt['real_total_posts'].mean():.1f} ± {pt['real_total_posts'].std():.1f}",
        "",
        "=" * 72,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)

    (OUTPUT_DIR / "summary.txt").write_text(summary_text + "\n")

    aggregate = {
        "n_threads": n,
        "n_tweets_total": len(df),
        "sentiment_jsd_mean": float(pt["sentiment_jsd"].mean()),
        "political_jsd_mean": float(pt["political_jsd"].mean()),
        "emotion_jsd_mean": float(pt["emotion_jsd"].mean()),
        "sentiment_residual_mean": float(pt["sentiment_residual"].mean()),
        "sentiment_residual_std": float(pt["sentiment_residual"].std()),
        "aggression_pearson_r": float(r_agg),
        "overall_accuracy_mean": float(acc.mean()),
        "overall_accuracy_std": float(acc.std()),
        "wilcoxon_p_value": float(p_val),
        "rating_excellent": n_exc,
        "rating_good": n_good,
    }
    with open(OUTPUT_DIR / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    pt = compute_per_thread_metrics(df)

    print("\nGenerating figures...")
    make_figures(pt, df)

    print()
    write_summary(pt, df)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
