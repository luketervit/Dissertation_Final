"""
Analyse ablation sweep results: classify all simulated tweets, compute
per-config metrics, and produce a ranked summary table.

Classifies ~15k tweets through 5 RoBERTa models (CPU ~2-3h, GPU ~20min).
Checkpoints after each config so progress is never lost.

Usage:
    python scripts/analyse_sweep.py
    python scripts/analyse_sweep.py --no-resume   # fresh start
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_comprehensive import (
    POLITICAL_LABEL_MAP,
    load_models,
    load_thread,
    compute_distribution,
    compute_jsd,
)
from validate_batch_100 import classify_tweet_full

# ── Paths ────────────────────────────────────────────────────────────────────

SWEEP_DIR = PROJECT_ROOT / "parameter_sweep"
REAL_DIR = PROJECT_ROOT / "batch_simulations_reconstructed"
OUTPUT_DIR = PROJECT_ROOT / "parameter_sweep" / "analysis"
CHECKPOINT_PATH = OUTPUT_DIR / "all_tweets_classified.csv"
PER_RUN_PATH = OUTPUT_DIR / "per_run_results.csv"
PER_CONFIG_PATH = OUTPUT_DIR / "per_config_summary.csv"
RANKING_PATH = OUTPUT_DIR / "config_ranking.txt"
FIGURES_DIR = OUTPUT_DIR / "figures"

THREADS = [1, 5, 10, 15, 20, 30, 50, 60, 75, 90]


# ── Classification ───────────────────────────────────────────────────────────


def classify_all(models: dict, resume: bool = True) -> pd.DataFrame:
    """Classify every tweet across all configs and threads."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    rows: list[dict] = []

    if resume and CHECKPOINT_PATH.exists():
        df_existing = pd.read_csv(CHECKPOINT_PATH)
        rows = df_existing.to_dict("records")
        already_done = set(df_existing["run_id"].unique())
        print(f"Resuming: {len(already_done)} runs already classified")

    # Also classify real threads (once, keyed as "REAL")
    real_done = any(r.get("config") == "REAL" for r in rows)

    if not real_done:
        print("\nClassifying real threads...")
        for t in tqdm(THREADS, desc="Real threads"):
            thread_name = f"thread_{t:03d}"
            real_path = REAL_DIR / thread_name / "thread_metadata.json"
            if not real_path.exists():
                print(f"  Skipping {thread_name}: no real data")
                continue
            real_tweets = load_thread(str(real_path))
            for tweet in real_tweets:
                dna = classify_tweet_full(tweet["text"], models)
                dna["config"] = "REAL"
                dna["thread"] = thread_name
                dna["run_id"] = f"REAL/{thread_name}"
                dna["source"] = "real"
                dna["text"] = tweet["text"]
                rows.append(dna)
        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)

    # Classify simulated tweets per config
    config_dirs = sorted(
        d for d in SWEEP_DIR.iterdir()
        if d.is_dir() and d.name != "analysis"
    )
    total_runs = sum(
        1 for cd in config_dirs
        for t in THREADS
        if (cd / f"thread_{t:03d}" / "simulation_output"
            / "simulated_thread_metadata.json").exists()
    )
    print(f"\n{len(config_dirs)} configs, {total_runs} total runs to classify")

    done_count = 0
    for config_dir in config_dirs:
        config_name = config_dir.name
        for t in THREADS:
            thread_name = f"thread_{t:03d}"
            run_id = f"{config_name}/{thread_name}"

            if run_id in already_done:
                done_count += 1
                continue

            sim_path = (
                config_dir / thread_name / "simulation_output"
                / "simulated_thread_metadata.json"
            )
            if not sim_path.exists():
                continue

            done_count += 1
            print(
                f"  [{done_count}/{total_runs}] {run_id}...",
                end="", flush=True,
            )

            try:
                sim_tweets = load_thread(str(sim_path))
            except Exception as exc:
                print(f" ERROR: {exc}")
                continue

            for tweet in sim_tweets:
                dna = classify_tweet_full(tweet["text"], models)
                dna["config"] = config_name
                dna["thread"] = thread_name
                dna["run_id"] = run_id
                dna["source"] = "simulated"
                dna["text"] = tweet["text"]
                rows.append(dna)

            already_done.add(run_id)
            pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
            n_sim = sum(1 for r in rows if r["run_id"] == run_id)
            print(f" {n_sim} tweets")

    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nClassification complete: {len(df)} total tweets")
    return df


# ── Per-run metrics ──────────────────────────────────────────────────────────


def compute_per_run_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute validation metrics for each (config, thread) pair."""
    real_by_thread: dict[str, pd.DataFrame] = {}
    for thread, tdf in df[df["source"] == "real"].groupby("thread"):
        real_by_thread[thread] = tdf

    records = []
    sim_df = df[df["source"] == "simulated"]

    for (config, thread), sdf in sim_df.groupby(["config", "thread"]):
        rdf = real_by_thread.get(thread)
        if rdf is None or len(rdf) == 0 or len(sdf) == 0:
            continue

        # Sentiment
        real_sent_mean = rdf["sentiment_continuous"].mean()
        sim_sent_mean = sdf["sentiment_continuous"].mean()

        real_sent_dist = compute_distribution(
            rdf.to_dict("records"), "sentiment_label"
        )
        sim_sent_dist = compute_distribution(
            sdf.to_dict("records"), "sentiment_label"
        )
        sent_jsd = compute_jsd(real_sent_dist, sim_sent_dist)

        # Political
        real_pol_dist = compute_distribution(
            rdf.to_dict("records"), "political_label"
        )
        sim_pol_dist = compute_distribution(
            sdf.to_dict("records"), "political_label"
        )
        pol_jsd = compute_jsd(real_pol_dist, sim_pol_dist)

        # Emotion
        real_emo_dist = compute_distribution(
            rdf.to_dict("records"), "emotion_label"
        )
        sim_emo_dist = compute_distribution(
            sdf.to_dict("records"), "emotion_label"
        )
        emo_jsd = compute_jsd(real_emo_dist, sim_emo_dist)

        # Aggression
        real_agg = (rdf["hate_score"] + rdf["offensive_score"]).mean()
        sim_agg = (sdf["hate_score"] + sdf["offensive_score"]).mean()

        # Negative %
        real_neg = (rdf["sentiment_label"] == "negative").mean()
        sim_neg = (sdf["sentiment_label"] == "negative").mean()

        # Right %
        real_right = (rdf["political_label"] == "Right").mean()
        sim_right = (sdf["political_label"] == "Right").mean()

        # Overall accuracy (same formula as batch validation)
        overall = (
            0.30 * (1 - pol_jsd)
            + 0.20 * (1 - emo_jsd)
            + 0.30 * (1 - sent_jsd)
            + 0.20 * (1 - abs(sim_agg - real_agg))
        ) * 100

        records.append({
            "config": config,
            "thread": thread,
            "sentiment_residual": sim_sent_mean - real_sent_mean,
            "sentiment_jsd": sent_jsd,
            "political_jsd": pol_jsd,
            "emotion_jsd": emo_jsd,
            "real_aggression": real_agg,
            "sim_aggression": sim_agg,
            "aggression_gap": abs(sim_agg - real_agg),
            "real_neg_pct": real_neg,
            "sim_neg_pct": sim_neg,
            "real_right_pct": real_right,
            "sim_right_pct": sim_right,
            "overall_accuracy": overall,
            "n_real": len(rdf),
            "n_sim": len(sdf),
        })

    per_run = pd.DataFrame(records)
    per_run.to_csv(PER_RUN_PATH, index=False)
    print(f"Per-run metrics: {len(per_run)} rows")
    return per_run


# ── Per-config aggregation ───────────────────────────────────────────────────


def compute_per_config_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run metrics to per-config means."""
    agg = per_run.groupby("config").agg(
        overall_mean=("overall_accuracy", "mean"),
        overall_std=("overall_accuracy", "std"),
        sent_residual_mean=("sentiment_residual", "mean"),
        sent_residual_std=("sentiment_residual", "std"),
        sent_jsd_mean=("sentiment_jsd", "mean"),
        pol_jsd_mean=("political_jsd", "mean"),
        emo_jsd_mean=("emotion_jsd", "mean"),
        agg_gap_mean=("aggression_gap", "mean"),
        sim_neg_mean=("sim_neg_pct", "mean"),
        real_neg_mean=("real_neg_pct", "mean"),
        sim_right_mean=("sim_right_pct", "mean"),
        n_threads=("thread", "count"),
    ).reset_index()

    # Sentiment bias significance per config
    p_values = []
    for config, cdf in per_run.groupby("config"):
        residuals = cdf["sentiment_residual"].values
        if len(residuals) >= 5:
            try:
                _, p = wilcoxon(residuals)
            except ValueError:
                p = 1.0
        else:
            p = np.nan
        p_values.append({"config": config, "wilcoxon_p": p})
    p_df = pd.DataFrame(p_values)
    agg = agg.merge(p_df, on="config")

    # Rating
    agg["rating"] = agg["overall_mean"].apply(
        lambda x: "EXCELLENT" if x >= 80
        else "GOOD" if x >= 60
        else "FAIR" if x >= 40
        else "POOR"
    )

    agg = agg.sort_values("overall_mean", ascending=False)
    agg.to_csv(PER_CONFIG_PATH, index=False)
    return agg


# ── Figures ──────────────────────────────────────────────────────────────────


def plot_figures(per_run: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Generate sweep analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    configs_ranked = summary["config"].tolist()

    # Fig 1: Overall accuracy by config (horizontal bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        "#2ecc71" if r == "EXCELLENT" else
        "#f39c12" if r == "GOOD" else
        "#e74c3c" for r in summary["rating"]
    ]
    ax.barh(
        range(len(configs_ranked)),
        summary["overall_mean"],
        xerr=summary["overall_std"],
        color=colors,
        edgecolor="white",
        capsize=3,
    )
    ax.set_yticks(range(len(configs_ranked)))
    ax.set_yticklabels(configs_ranked, fontsize=9)
    ax.set_xlabel("Overall Accuracy (%)")
    ax.set_title("Ablation Sweep: Overall Accuracy by Configuration")
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="EXCELLENT threshold")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_accuracy_ranking.png", dpi=300)
    plt.close()

    # Fig 2: Sentiment residual by config (box plot)
    fig, ax = plt.subplots(figsize=(10, 8))
    data_by_config = [
        per_run[per_run["config"] == c]["sentiment_residual"].values
        for c in configs_ranked
    ]
    bp = ax.boxplot(
        data_by_config, vert=False, labels=configs_ranked,
        patch_artist=True, widths=0.6,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)
    ax.axvline(x=0, color="red", linestyle="--", label="Zero bias")
    ax.set_xlabel("Sentiment Residual (sim - real)")
    ax.set_title("Sentiment Bias by Configuration")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_sentiment_residual.png", dpi=300)
    plt.close()

    # Fig 3: JSD heatmap (configs × dimensions)
    jsd_cols = ["sent_jsd_mean", "pol_jsd_mean", "emo_jsd_mean"]
    jsd_labels = ["Sentiment", "Political", "Emotion"]
    jsd_data = summary.set_index("config")[jsd_cols].values

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(jsd_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.25)
    ax.set_xticks(range(len(jsd_labels)))
    ax.set_xticklabels(jsd_labels)
    ax.set_yticks(range(len(configs_ranked)))
    ax.set_yticklabels(configs_ranked, fontsize=9)
    for i in range(len(configs_ranked)):
        for j in range(len(jsd_labels)):
            ax.text(j, i, f"{jsd_data[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="JSD (lower = better)")
    ax.set_title("JSD Heatmap: Configs × Dimensions")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_jsd_heatmap.png", dpi=300)
    plt.close()

    # Fig 4: Aggression gap by config
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(configs_ranked)),
        summary["agg_gap_mean"],
        color="#e67e22",
        edgecolor="white",
    )
    ax.set_yticks(range(len(configs_ranked)))
    ax.set_yticklabels(configs_ranked, fontsize=9)
    ax.set_xlabel("Mean |Aggression Gap| (sim - real)")
    ax.set_title("Aggression Calibration by Configuration")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_aggression_gap.png", dpi=300)
    plt.close()

    # Fig 5: Top 5 vs bottom 5 comparison radar-ish grouped bars
    top5 = summary.head(5)
    bot5 = summary.tail(5)
    metrics = ["overall_mean", "sent_jsd_mean", "pol_jsd_mean", "emo_jsd_mean", "agg_gap_mean"]
    metric_labels = ["Accuracy", "Sent JSD", "Pol JSD", "Emo JSD", "Agg Gap"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (subset, title) in enumerate([(top5, "Top 5"), (bot5, "Bottom 5")]):
        x = np.arange(len(metrics))
        width = 0.15
        for i, (_, row) in enumerate(subset.iterrows()):
            vals = [row[m] for m in metrics]
            # Normalize: accuracy /100, JSDs and gap are already 0-1 scale
            vals[0] = vals[0] / 100
            axes[ax_i].bar(x + i * width, vals, width, label=row["config"])
        axes[ax_i].set_xticks(x + width * 2)
        axes[ax_i].set_xticklabels(metric_labels, fontsize=9)
        axes[ax_i].set_title(title)
        axes[ax_i].legend(fontsize=7, loc="upper right")
        axes[ax_i].set_ylim(0, 1.1)
    plt.suptitle("Top 5 vs Bottom 5 Configurations")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_top_vs_bottom.png", dpi=300)
    plt.close()

    print(f"Saved 5 figures to {FIGURES_DIR}/")


# ── Summary report ───────────────────────────────────────────────────────────


def write_ranking(summary: pd.DataFrame) -> None:
    """Write human-readable ranking to text file and stdout."""
    lines = []
    lines.append("=" * 80)
    lines.append("ABLATION SWEEP RESULTS — 21 CONFIGS × 10 THREADS (0-SHOT)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Rank':<5} {'Config':<28} {'Accuracy':>10} {'Sent JSD':>10} "
                 f"{'Pol JSD':>10} {'Emo JSD':>10} {'Agg Gap':>10} "
                 f"{'Sent Res':>10} {'Wilcox p':>10} {'Rating':<10}")
    lines.append("-" * 120)

    for rank, (_, row) in enumerate(summary.iterrows(), 1):
        lines.append(
            f"{rank:<5} {row['config']:<28} "
            f"{row['overall_mean']:>9.1f}% "
            f"{row['sent_jsd_mean']:>10.4f} "
            f"{row['pol_jsd_mean']:>10.4f} "
            f"{row['emo_jsd_mean']:>10.4f} "
            f"{row['agg_gap_mean']:>10.4f} "
            f"{row['sent_residual_mean']:>+10.3f} "
            f"{row['wilcoxon_p']:>10.4f} "
            f"{row['rating']:<10}"
        )

    lines.append("")
    lines.append("=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)

    best = summary.iloc[0]
    worst = summary.iloc[-1]
    baseline = summary[summary["config"] == "0shot_baseline"]

    lines.append(f"\nBest config:  {best['config']} — {best['overall_mean']:.1f}%")
    lines.append(f"Worst config: {worst['config']} — {worst['overall_mean']:.1f}%")
    if len(baseline) > 0:
        bl = baseline.iloc[0]
        lines.append(f"0-shot baseline: {bl['overall_mean']:.1f}%")
        lines.append(f"Best vs baseline: {best['overall_mean'] - bl['overall_mean']:+.1f}%")

    # Which parameters help vs hurt
    lines.append("\n" + "-" * 80)
    lines.append("PARAMETER IMPACT (vs 0-shot baseline)")
    lines.append("-" * 80)

    if len(baseline) > 0:
        bl_acc = baseline.iloc[0]["overall_mean"]
        for _, row in summary.iterrows():
            if row["config"] == "0shot_baseline":
                continue
            delta = row["overall_mean"] - bl_acc
            direction = "+" if delta > 0 else ""
            lines.append(
                f"  {row['config']:<28} {direction}{delta:.1f}%  "
                f"(sent_res={row['sent_residual_mean']:+.3f})"
            )

    # Unbiased configs (Wilcoxon p > 0.05)
    unbiased = summary[summary["wilcoxon_p"] > 0.05]
    lines.append(f"\nConfigs with NO significant sentiment bias (p>0.05): "
                 f"{len(unbiased)}/{len(summary)}")
    for _, row in unbiased.iterrows():
        lines.append(f"  {row['config']:<28} p={row['wilcoxon_p']:.4f}")

    text = "\n".join(lines)
    with open(RANKING_PATH, "w") as f:
        f.write(text)
    print(text)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyse ablation sweep")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip classification, use existing checkpoint")
    args = parser.parse_args()

    if args.skip_classify and CHECKPOINT_PATH.exists():
        print(f"Loading existing classifications from {CHECKPOINT_PATH}")
        df = pd.read_csv(CHECKPOINT_PATH)
    else:
        models = load_models(device=args.device)
        df = classify_all(models, resume=not args.no_resume)

    per_run = compute_per_run_metrics(df)
    summary = compute_per_config_summary(per_run)
    plot_figures(per_run, summary)
    write_ranking(summary)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
