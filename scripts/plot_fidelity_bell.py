#!/usr/bin/env python3
"""Clean bell-curve (KDE + histogram) of per-thread fidelity scores."""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from pathlib import Path

CSV  = Path(__file__).resolve().parent.parent / "batch_analysis_new" / "fidelity_scores.csv"
OUT  = Path(__file__).resolve().parent.parent / "batch_analysis_new" / "fidelity_bell_curve.png"

df = pd.read_csv(CSV)
scores = df["fidelity"].values
mu, sigma = scores.mean(), scores.std()

fig, ax = plt.subplots(figsize=(9, 5))

# histogram (density-normalised)
ax.hist(scores, bins=18, density=True, color="#3498db", alpha=0.45,
        edgecolor="white", label="Observed")

# KDE (actual shape)
xs = np.linspace(0.3, 1.02, 300)
kde = gaussian_kde(scores, bw_method=0.25)
ax.plot(xs, kde(xs), color="#2c3e50", lw=2.5, label="KDE (actual shape)")

# fitted normal
ax.plot(xs, norm.pdf(xs, mu, sigma), color="#e74c3c", lw=2, ls="--",
        label=f"Normal fit (μ={mu:.3f}, σ={sigma:.3f})")

# mean line
ax.axvline(mu, color="#e74c3c", ls=":", lw=1.5, alpha=0.7)

# annotate
ax.annotate(f"μ = {mu:.3f}", xy=(mu, ax.get_ylim()[1]*0.92),
            fontsize=12, fontweight="bold", color="#e74c3c",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.85))

ax.set_xlabel("Composite Fidelity Score", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Simulation Fidelity Distribution  (n = 100 threads)", fontsize=15, fontweight="bold")
ax.set_xlim(0.35, 1.02)
ax.legend(fontsize=11, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT}")
