"""
Batch Validation of 100 Thread Simulations

Classifies all tweets (real + simulated) across 100 threads through 5 RoBERTa
models, computes per-thread metrics including thread structure (depth,
subthreads), and generates 8 dissertation-quality figures.

Usage:
    # Old params
    python scripts/validate_batch_old_params.py --variant old [--device auto|cuda|cpu]
    # New params
    python scripts/validate_batch_old_params.py --variant new [--device auto|cuda|cpu]
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

REAL_DIR = PROJECT_ROOT / "batch_simulations_reconstructed"
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
POLITICAL_LABELS = ["Left", "Center", "Right"]


def _setup_paths(variant: str) -> None:
    """Set global paths based on variant (old/new)."""
    global SIM_DIR, OUTPUT_DIR, FIGURES_DIR, CHECKPOINT_PATH
    global PER_THREAD_PATH, AGGREGATE_PATH, SUMMARY_PATH, VARIANT_LABEL

    VARIANT_LABEL = "Old Params" if variant == "old" else "New Params"
    SIM_DIR = PROJECT_ROOT / f"batch_output_{variant}"
    OUTPUT_DIR = PROJECT_ROOT / f"batch_analysis_{variant}"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    CHECKPOINT_PATH = OUTPUT_DIR / "all_tweets_classified.csv"
    PER_THREAD_PATH = OUTPUT_DIR / "per_thread_results.csv"
    AGGREGATE_PATH = OUTPUT_DIR / "aggregate_results.json"
    SUMMARY_PATH = OUTPUT_DIR / "summary.txt"


# ── Classification ───────────────────────────────────────────────────────────


def classify_tweet_full(text: str, models: dict) -> dict:
    """Classify a tweet through all 5 models with full sentiment probs."""
    text = str(text)[:512]

    p = models["political"](text)[0]
    e = models["emotion"](text)[0]
    h = models["hate"](text)[0]
    o = models["offensive"](text)[0]

    sn_all = models["sentiment"](text, top_k=None)
    sn_probs = {r["label"]: r["score"] for r in sn_all}
    sn_top = max(sn_all, key=lambda x: x["score"])

    p_pos = sn_probs.get("positive", 0.0)
    p_neg = sn_probs.get("negative", 0.0)

    return {
        "political_label": POLITICAL_LABEL_MAP.get(p["label"], p["label"]),
        "political_score": p["score"],
        "emotion_label": e["label"],
        "emotion_score": e["score"],
        "sentiment_label": sn_top["label"],
        "sentiment_score": sn_top["score"],
        "sentiment_pos": p_pos,
        "sentiment_neg": p_neg,
        "sentiment_neu": sn_probs.get("neutral", 0.0),
        "sentiment_continuous": p_pos - p_neg,
        "hate_score": h["score"] if h["label"] == "HATE" else 1 - h["score"],
        "offensive_score": (
            o["score"] if o["label"] == "OFFENSIVE" else 1 - o["score"]
        ),
    }


# ── Thread structure helpers ─────────────────────────────────────────────────


def compute_thread_structure(history: list[dict]) -> dict:
    """Compute depth and subthread metrics from thread_history.json data.

    Returns:
        max_depth: deepest reply chain
        mean_depth: average depth of all posts
        num_subthreads: number of distinct reply chains off the root
        depth_distribution: {depth: count}
        total_posts: number of posts (incl root)
    """
    if not history:
        return {
            "max_depth": 0, "mean_depth": 0.0, "num_subthreads": 0,
            "depth_distribution": {}, "total_posts": 0,
        }

    depths = [post.get("depth", 0) for post in history]
    # Subthreads = distinct posts at depth 1 (direct replies to root)
    num_subthreads = sum(1 for post in history if post.get("depth") == 1)
    depth_counts = dict(Counter(depths))

    return {
        "max_depth": max(depths),
        "mean_depth": float(np.mean(depths)),
        "num_subthreads": num_subthreads,
        "depth_distribution": depth_counts,
        "total_posts": len(history),
    }


def compute_real_thread_structure(events: list[dict]) -> dict:
    """Compute structure for real threads (flat — all replies to root).

    Real threads from the USC dataset don't have parent_id info, so all
    non-root tweets are assumed to be depth-1 replies to root.
    """
    n = len(events)
    root_count = sum(1 for e in events if e.get("is_root"))
    replies = n - root_count

    return {
        "max_depth": 1 if replies > 0 else 0,
        "mean_depth": replies / n if n > 0 else 0.0,
        "num_subthreads": replies,  # each reply is its own "subthread"
        "depth_distribution": {0: root_count, 1: replies} if replies > 0 else {0: root_count},
        "total_posts": n,
    }


# ── Step 1: Classify all tweets ─────────────────────────────────────────────


def classify_all_threads(models: dict, resume: bool = True) -> pd.DataFrame:
    """Classify every tweet in all 100 threads with checkpoint recovery."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    rows: list[dict] = []
    if resume and CHECKPOINT_PATH.exists():
        df_existing = pd.read_csv(CHECKPOINT_PATH)
        rows = df_existing.to_dict("records")
        already_done = set(df_existing["thread_id"].unique())
        print(f"Resuming from checkpoint: {len(already_done)} threads done")

    # Discover threads that exist in BOTH real and sim dirs
    real_threads = {
        d.name for d in REAL_DIR.iterdir()
        if d.is_dir() and d.name.startswith("thread_")
    }
    sim_threads = {
        d.name for d in SIM_DIR.iterdir()
        if d.is_dir() and d.name.startswith("thread_")
    }
    common = sorted(real_threads & sim_threads)
    print(f"Found {len(common)} threads with both real and simulated data")

    for thread_id in tqdm(common, desc="Threads"):
        if thread_id in already_done:
            continue

        real_path = REAL_DIR / thread_id / "thread_metadata.json"
        sim_path = SIM_DIR / thread_id / "simulation_output" / "simulated_thread_metadata.json"

        if not real_path.exists() or not sim_path.exists():
            print(f"  Skipping {thread_id}: missing files")
            continue

        try:
            real_tweets = load_thread(str(real_path))
            sim_tweets = load_thread(str(sim_path))
        except Exception as exc:
            print(f"  Skipping {thread_id}: {exc}")
            continue

        for tweet in real_tweets:
            dna = classify_tweet_full(tweet["text"], models)
            dna["thread_id"] = thread_id
            dna["source"] = "real"
            dna["text"] = tweet["text"]
            dna["tweet_id"] = tweet.get("tweet_id")
            rows.append(dna)

        for tweet in sim_tweets:
            dna = classify_tweet_full(tweet["text"], models)
            dna["thread_id"] = thread_id
            dna["source"] = "simulated"
            dna["text"] = tweet["text"]
            dna["tweet_id"] = tweet.get("tweet_id")
            rows.append(dna)

        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
        already_done.add(thread_id)

    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"Classification complete: {len(df)} tweets, {df['thread_id'].nunique()} threads")
    return df


# ── Step 2: Per-thread metrics ───────────────────────────────────────────────


def compute_per_thread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute validation metrics + thread structure for each thread."""
    records = []

    # Discover threads
    real_threads = {
        d.name for d in REAL_DIR.iterdir()
        if d.is_dir() and d.name.startswith("thread_")
    }

    for thread_id, tdf in df.groupby("thread_id"):
        real = tdf[tdf["source"] == "real"]
        sim = tdf[tdf["source"] == "simulated"]

        if len(real) == 0 or len(sim) == 0:
            continue

        # ── Sentiment ────────────────────────────────────────────────
        real_sent_mean = real["sentiment_continuous"].mean()
        sim_sent_mean = sim["sentiment_continuous"].mean()

        real_sent_dist = compute_distribution(real.to_dict("records"), "sentiment_label")
        sim_sent_dist = compute_distribution(sim.to_dict("records"), "sentiment_label")
        sent_jsd = compute_jsd(real_sent_dist, sim_sent_dist)

        # ── Political ────────────────────────────────────────────────
        real_pol_dist = compute_distribution(real.to_dict("records"), "political_label")
        sim_pol_dist = compute_distribution(sim.to_dict("records"), "political_label")
        pol_jsd = compute_jsd(real_pol_dist, sim_pol_dist)

        # ── Emotion ──────────────────────────────────────────────────
        real_emo_dist = compute_distribution(real.to_dict("records"), "emotion_label")
        sim_emo_dist = compute_distribution(sim.to_dict("records"), "emotion_label")
        emo_jsd = compute_jsd(real_emo_dist, sim_emo_dist)

        # ── Aggression ───────────────────────────────────────────────
        real_agg = (real["hate_score"] + real["offensive_score"]).mean()
        sim_agg = (sim["hate_score"] + sim["offensive_score"]).mean()

        # ── Political percentages ────────────────────────────────────
        real_right_pct = real_pol_dist.get("Right", 0.0)
        sim_right_pct = sim_pol_dist.get("Right", 0.0)
        real_neg_pct = real_sent_dist.get("negative", 0.0)
        sim_neg_pct = sim_sent_dist.get("negative", 0.0)

        # ── Overall accuracy ─────────────────────────────────────────
        pol_sim = (1 - pol_jsd) * 100
        emo_sim = (1 - emo_jsd) * 100
        sent_sim = (1 - sent_jsd) * 100
        agg_sim = max((1 - abs(real_agg - sim_agg)) * 100, 0.0)

        overall = pol_sim * 0.30 + emo_sim * 0.20 + sent_sim * 0.30 + agg_sim * 0.20

        # ── Thread structure ─────────────────────────────────────────
        # Simulated: load thread_history.json for parent/depth data
        sim_history_path = (
            SIM_DIR / str(thread_id) / "simulation_output" / "thread_history.json"
        )
        if sim_history_path.exists():
            with open(sim_history_path) as f:
                sim_history = json.load(f)
            sim_struct = compute_thread_structure(sim_history)
        else:
            sim_struct = {
                "max_depth": 0, "mean_depth": 0.0,
                "num_subthreads": 0, "total_posts": len(sim),
            }

        # Real: load from thread_metadata.json (flat structure)
        real_meta_path = REAL_DIR / str(thread_id) / "thread_metadata.json"
        if real_meta_path.exists():
            with open(real_meta_path) as f:
                real_meta = json.load(f)
            real_events = real_meta.get("temporal_events", [])
            real_struct = compute_real_thread_structure(real_events)
        else:
            real_struct = {
                "max_depth": 1, "mean_depth": 0.0,
                "num_subthreads": len(real), "total_posts": len(real),
            }

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
            "real_neg_pct": real_neg_pct,
            "sim_neg_pct": sim_neg_pct,
            "overall_accuracy": overall,
            # Structure metrics
            "real_total_posts": real_struct["total_posts"],
            "sim_total_posts": sim_struct["total_posts"],
            "real_max_depth": real_struct["max_depth"],
            "sim_max_depth": sim_struct["max_depth"],
            "real_mean_depth": real_struct["mean_depth"],
            "sim_mean_depth": sim_struct["mean_depth"],
            "real_num_subthreads": real_struct["num_subthreads"],
            "sim_num_subthreads": sim_struct["num_subthreads"],
        })

    result = pd.DataFrame(records)
    result.to_csv(PER_THREAD_PATH, index=False)
    print(f"Per-thread metrics saved: {PER_THREAD_PATH} ({len(result)} threads)")
    return result


# ── Step 3: Figures ──────────────────────────────────────────────────────────


def make_figures(per_thread: pd.DataFrame, all_tweets: pd.DataFrame) -> None:
    """Generate 8 dissertation-quality figures."""
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
    _fig7_depth_comparison(per_thread)
    _fig8_subthread_comparison(per_thread)

    print(f"All 8 figures saved to {FIGURES_DIR}/")


def _fig1_sentiment_residual(pt: pd.DataFrame) -> None:
    residuals = pt["sentiment_residual"].values
    mu, sigma = residuals.mean(), residuals.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=25, edgecolor="black", alpha=0.75, color="#4CAF50")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect match")
    ax.axvline(mu, color="navy", linestyle="-", linewidth=1.5, label=f"Mean = {mu:.3f}")
    ax.set_xlabel("Sentiment Residual (simulated − real)")
    ax.set_ylabel("Number of Threads")
    ax.set_title(f"Distribution of Sentiment Residuals Across 100 Threads ({VARIANT_LABEL})")
    ax.legend()
    ax.text(0.97, 0.95, f"μ = {mu:.3f}\nσ = {sigma:.3f}\nn = {len(residuals)}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_sentiment_residual_histogram.png", dpi=300)
    plt.close(fig)
    print("  Fig 1: sentiment residual histogram")


def _fig2_jsd_boxplot(pt: pd.DataFrame) -> None:
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
    ax.set_title(f"Validation Quality Across 100 Threads ({VARIANT_LABEL})")
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


def _fig3_aggression_scatter(pt: pd.DataFrame) -> None:
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


def _fig4_overall_accuracy_histogram(pt: pd.DataFrame) -> None:
    acc = pt["overall_accuracy"].values
    mu, sigma = acc.mean(), acc.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(acc, bins=20, edgecolor="black", alpha=0.75, color="#FF9800")
    for val, label, colour in [
        (80, "EXCELLENT (≥80%)", "#4CAF50"),
        (60, "GOOD (≥60%)", "#2196F3"),
        (40, "FAIR (≥40%)", "#FFC107"),
    ]:
        ax.axvline(val, color=colour, linestyle="--", linewidth=1.2, label=label)

    ax.set_xlabel("Overall Weighted Accuracy (%)")
    ax.set_ylabel("Number of Threads")
    ax.set_title(f"Distribution of Overall Accuracy ({VARIANT_LABEL})")
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
    ax.set_title(f"Aggregate Political Distribution ({VARIANT_LABEL})")
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


def _fig7_depth_comparison(pt: pd.DataFrame) -> None:
    """Scatter plot: real vs simulated max thread depth."""
    x = pt["real_max_depth"].values.astype(float)
    y = pt["sim_max_depth"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Jitter for visibility (real threads are all depth 1)
    x_jitter = x + np.random.normal(0, 0.08, len(x))
    ax.scatter(x_jitter, y, alpha=0.6, s=50, edgecolors="black", linewidths=0.5,
               color="#4CAF50")

    # Reference line
    max_val = max(x.max(), y.max()) + 1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")

    ax.set_xlabel("Real Thread Max Depth")
    ax.set_ylabel("Simulated Thread Max Depth")
    ax.set_title(f"Thread Depth: Real vs Simulated ({VARIANT_LABEL})")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.grid(alpha=0.3)

    # Stats
    sim_mean = y.mean()
    sim_std = y.std()
    ax.text(0.97, 0.05,
            f"Real max depth: all = 1 (flat)\n"
            f"Sim max depth: {sim_mean:.1f} ± {sim_std:.1f}\n"
            f"Sim range: [{int(y.min())}, {int(y.max())}]",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_depth_comparison.png", dpi=300)
    plt.close(fig)
    print("  Fig 7: depth comparison")


def _fig8_subthread_comparison(pt: pd.DataFrame) -> None:
    """Scatter plot: real vs simulated number of subthreads (depth-1 replies)."""
    x = pt["real_num_subthreads"].values.astype(float)
    y = pt["sim_num_subthreads"].values.astype(float)

    r_val, p_val = pearsonr(x, y) if len(x) > 2 else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.6, s=50, edgecolors="black", linewidths=0.5,
               color="#9C27B0")

    max_val = max(x.max(), y.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")

    ax.set_xlabel("Real Thread Subthreads (direct replies)")
    ax.set_ylabel("Simulated Thread Subthreads (depth-1 replies)")
    ax.set_title(f"Subthread Count: Real vs Simulated ({VARIANT_LABEL})")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(alpha=0.3)

    ax.text(0.97, 0.05,
            f"Pearson r = {r_val:.3f}\np = {p_val:.2e}\n"
            f"Real mean: {x.mean():.1f}\nSim mean: {y.mean():.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_subthread_comparison.png", dpi=300)
    plt.close(fig)
    print("  Fig 8: subthread comparison")


# ── Step 4: Summary statistics ───────────────────────────────────────────────


def write_summary(per_thread: pd.DataFrame, all_tweets: pd.DataFrame) -> None:
    """Write human-readable summary + aggregate JSON."""
    n = len(per_thread)
    acc = per_thread["overall_accuracy"].values

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

    # Aggression correlation
    r_agg, p_agg = pearsonr(
        per_thread["real_aggression_mean"].values,
        per_thread["sim_aggression_mean"].values,
    )

    lines = [
        "=" * 72,
        f"  BATCH VALIDATION SUMMARY — 100 THREADS ({VARIANT_LABEL.upper()})",
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
        f"    Aggression Pearson r: {r_agg:.3f} (p={p_agg:.2e})",
        f"    Overall accuracy:   {acc.mean():.1f}% ± {acc.std():.1f}%",
        "",
        "  Rating Breakdown:",
        f"    EXCELLENT (≥80%): {n_exc}",
        f"    GOOD (60–80%):    {n_good}",
        f"    FAIR (40–60%):    {n_fair}",
        f"    POOR (<40%):      {n_poor}",
        "",
        "  Wilcoxon Signed-Rank Test (per-thread mean sentiment):",
        f"    Statistic: {stat:.2f}",
        f"    p-value:   {p_val:.4e}",
        f"    Significant (α=0.05): {'Yes' if p_val < 0.05 else 'No'}",
        "",
        "  Thread Structure (Simulated):",
        f"    Max depth (mean ± std):    {per_thread['sim_max_depth'].mean():.1f} ± {per_thread['sim_max_depth'].std():.1f}",
        f"    Max depth range:           [{per_thread['sim_max_depth'].min()}, {per_thread['sim_max_depth'].max()}]",
        f"    Mean depth (mean ± std):   {per_thread['sim_mean_depth'].mean():.2f} ± {per_thread['sim_mean_depth'].std():.2f}",
        f"    Subthreads (mean ± std):   {per_thread['sim_num_subthreads'].mean():.1f} ± {per_thread['sim_num_subthreads'].std():.1f}",
        f"    Total posts (mean ± std):  {per_thread['sim_total_posts'].mean():.1f} ± {per_thread['sim_total_posts'].std():.1f}",
        "",
        "  Thread Structure (Real):",
        f"    All real threads are flat (max depth = 1)",
        f"    Subthreads (mean ± std):   {per_thread['real_num_subthreads'].mean():.1f} ± {per_thread['real_num_subthreads'].std():.1f}",
        f"    Total posts (mean ± std):  {per_thread['real_total_posts'].mean():.1f} ± {per_thread['real_total_posts'].std():.1f}",
        "",
        "=" * 72,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)

    SUMMARY_PATH.write_text(summary_text + "\n")
    print(f"\nSaved: {SUMMARY_PATH}")

    # Aggregate JSON
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
        "aggression_pearson_p": float(p_agg),
        "overall_accuracy_mean": float(acc.mean()),
        "overall_accuracy_std": float(acc.std()),
        "rating_excellent": n_exc,
        "rating_good": n_good,
        "rating_fair": n_fair,
        "rating_poor": n_poor,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_val),
        # Thread structure
        "sim_max_depth_mean": float(per_thread["sim_max_depth"].mean()),
        "sim_max_depth_std": float(per_thread["sim_max_depth"].std()),
        "sim_mean_depth_mean": float(per_thread["sim_mean_depth"].mean()),
        "sim_subthreads_mean": float(per_thread["sim_num_subthreads"].mean()),
        "sim_total_posts_mean": float(per_thread["sim_total_posts"].mean()),
        "real_subthreads_mean": float(per_thread["real_num_subthreads"].mean()),
        "real_total_posts_mean": float(per_thread["real_total_posts"].mean()),
    }
    with open(AGGREGATE_PATH, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved: {AGGREGATE_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch validate 100 thread simulations"
    )
    parser.add_argument("--variant", default="old", choices=["old", "new"],
                        help="Which param set to validate (old or new)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore checkpoint")
    args = parser.parse_args()

    _setup_paths(args.variant)
    resume = not args.no_resume

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Classify
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

    print(f"\nDone! All outputs in batch_analysis_{args.variant}/")


if __name__ == "__main__":
    main()
