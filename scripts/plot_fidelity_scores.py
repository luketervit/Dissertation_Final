#!/usr/bin/env python3
"""
Compute a single composite fidelity score (0-1) for each simulated thread
and plot all 100 to visualize variance.

Fidelity Score = weighted average of normalised sub-metrics:
  - Sentiment JSD  (lower = better, weight 0.30)
  - Emotion JSD    (lower = better, weight 0.20)
  - Political JSD  (lower = better, weight 0.15)
  - |Sentiment Residual| (lower = better, weight 0.20)
  - Overall Accuracy / 100 (higher = better, weight 0.15)

Each divergence metric is converted to a 0-1 fidelity by: 1 - min(value / cap, 1)
where cap is a reasonable upper bound (JSD max is ln2 ≈ 0.693, residual capped at 1).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------- config ----------
CSV_PATH = Path(__file__).resolve().parent.parent / "batch_analysis_new" / "per_thread_results.csv"
OUT_DIR  = Path(__file__).resolve().parent.parent / "batch_analysis_new"

WEIGHTS = {
    "sentiment_jsd":  0.30,
    "emotion_jsd":    0.20,
    "political_jsd":  0.15,
    "abs_residual":   0.20,
    "accuracy":       0.15,
}
JSD_CAP      = 0.693   # ln(2), theoretical max for JSD
RESIDUAL_CAP = 1.0     # sentiment residual capped at ±1

# ---------- load ----------
df = pd.read_csv(CSV_PATH)

# ---------- compute sub-scores (all 0-1, higher = better) ----------
df["f_sentiment_jsd"]  = 1 - np.clip(df["sentiment_jsd"]  / JSD_CAP, 0, 1)
df["f_emotion_jsd"]    = 1 - np.clip(df["emotion_jsd"]    / JSD_CAP, 0, 1)
df["f_political_jsd"]  = 1 - np.clip(df["political_jsd"]  / JSD_CAP, 0, 1)
df["f_abs_residual"]   = 1 - np.clip(df["sentiment_residual"].abs() / RESIDUAL_CAP, 0, 1)
df["f_accuracy"]       = np.clip(df["overall_accuracy"] / 100, 0, 1)

# ---------- composite ----------
df["fidelity"] = (
    WEIGHTS["sentiment_jsd"]  * df["f_sentiment_jsd"]  +
    WEIGHTS["emotion_jsd"]    * df["f_emotion_jsd"]    +
    WEIGHTS["political_jsd"]  * df["f_political_jsd"]  +
    WEIGHTS["abs_residual"]   * df["f_abs_residual"]    +
    WEIGHTS["accuracy"]       * df["f_accuracy"]
)

# ---------- stats ----------
mean = df["fidelity"].mean()
std  = df["fidelity"].std()
med  = df["fidelity"].median()
mn   = df["fidelity"].min()
mx   = df["fidelity"].max()

print(f"Fidelity Score  (n={len(df)})")
print(f"  Mean:   {mean:.4f}")
print(f"  Median: {med:.4f}")
print(f"  Std:    {std:.4f}")
print(f"  Min:    {mn:.4f}  (thread {df.loc[df['fidelity'].idxmin(), 'thread_id']})")
print(f"  Max:    {mx:.4f}  (thread {df.loc[df['fidelity'].idxmax(), 'thread_id']})")
print(f"  ≥0.90:  {(df['fidelity'] >= 0.90).sum()}")
print(f"  ≥0.80:  {(df['fidelity'] >= 0.80).sum()}")
print(f"  <0.70:  {(df['fidelity'] <  0.70).sum()}")

# ---------- FIGURE 1: bar chart sorted by fidelity ----------
fig, ax = plt.subplots(figsize=(14, 5))

df_sorted = df.sort_values("fidelity", ascending=False).reset_index(drop=True)

colours = []
for v in df_sorted["fidelity"]:
    if v >= 0.90:
        colours.append("#2ecc71")   # green – excellent
    elif v >= 0.80:
        colours.append("#3498db")   # blue  – good
    elif v >= 0.70:
        colours.append("#f39c12")   # amber – fair
    else:
        colours.append("#e74c3c")   # red   – poor

ax.bar(range(len(df_sorted)), df_sorted["fidelity"], color=colours, width=1.0, edgecolor="none")
ax.axhline(mean, color="#2c3e50", ls="--", lw=1.5, label=f"Mean = {mean:.3f}")
ax.axhline(mean - std, color="#95a5a6", ls=":", lw=1, label=f"±1 SD = {std:.3f}")
ax.axhline(mean + std, color="#95a5a6", ls=":", lw=1)
ax.fill_between(range(len(df_sorted)), mean - std, mean + std, alpha=0.08, color="#2c3e50")

ax.set_xlabel("Thread (ranked best → worst)", fontsize=12)
ax.set_ylabel("Composite Fidelity Score", fontsize=12)
ax.set_title("Per-Thread Simulation Fidelity  (n = 100 threads)", fontsize=14, fontweight="bold")
ax.set_xlim(-0.5, len(df_sorted) - 0.5)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(fontsize=10, loc="lower left")

# Colour legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="Excellent (≥0.90)"),
    Patch(facecolor="#3498db", label="Good (0.80–0.89)"),
    Patch(facecolor="#f39c12", label="Fair (0.70–0.79)"),
    Patch(facecolor="#e74c3c", label="Poor (<0.70)"),
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:2],
          fontsize=9, loc="lower left", ncol=3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fidelity_scores_ranked.png", dpi=200, bbox_inches="tight")
print(f"\nSaved → {OUT_DIR / 'fidelity_scores_ranked.png'}")

# ---------- FIGURE 2: histogram ----------
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(df["fidelity"], bins=15, color="#3498db", edgecolor="white", alpha=0.85)
ax2.axvline(mean, color="#e74c3c", ls="--", lw=2, label=f"Mean = {mean:.3f}")
ax2.axvline(med, color="#2ecc71", ls="--", lw=2, label=f"Median = {med:.3f}")
ax2.set_xlabel("Composite Fidelity Score", fontsize=12)
ax2.set_ylabel("Number of Threads", fontsize=12)
ax2.set_title("Distribution of Fidelity Scores  (n = 100)", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.set_xlim(0.4, 1.02)
plt.tight_layout()
fig2.savefig(OUT_DIR / "fidelity_histogram.png", dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'fidelity_histogram.png'}")

# ---------- FIGURE 3: sub-metric breakdown box plot ----------
fig3, ax3 = plt.subplots(figsize=(8, 5))
sub_cols = ["f_sentiment_jsd", "f_emotion_jsd", "f_political_jsd", "f_abs_residual", "f_accuracy"]
labels   = ["Sentiment\nJSD", "Emotion\nJSD", "Political\nJSD", "|Sentiment\nResidual|", "Overall\nAccuracy"]
bp = ax3.boxplot([df[c] for c in sub_cols], labels=labels, patch_artist=True,
                 boxprops=dict(facecolor="#3498db", alpha=0.6),
                 medianprops=dict(color="#e74c3c", lw=2))
ax3.set_ylabel("Sub-Score (0 = worst, 1 = best)", fontsize=12)
ax3.set_title("Sub-Metric Fidelity Breakdown  (n = 100)", fontsize=14, fontweight="bold")
ax3.set_ylim(0, 1.05)
plt.tight_layout()
fig3.savefig(OUT_DIR / "fidelity_sub_metrics.png", dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_DIR / 'fidelity_sub_metrics.png'}")

# ---------- save CSV ----------
out_csv = OUT_DIR / "fidelity_scores.csv"
df[["thread_id", "fidelity", "f_sentiment_jsd", "f_emotion_jsd", "f_political_jsd",
    "f_abs_residual", "f_accuracy"]].to_csv(out_csv, index=False, float_format="%.4f")
print(f"Saved → {out_csv}")
