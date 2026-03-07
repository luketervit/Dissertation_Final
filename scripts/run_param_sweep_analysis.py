#!/usr/bin/env python3
"""
Parameter Sweep Analysis
========================
Computes 8 simulation-fidelity metrics for every condition in the parameter
sweep and produces per-condition figures + a composite leaderboard.

Metrics
-------
1. SENTIMENT_JSD       – Jensen-Shannon divergence on VADER pos/neg/neu buckets
2. TEMPORAL_DECAY      – |real decay exponent − sim decay exponent| (power-law)
3. LEXICAL_DIVERSITY   – |real TTR − sim TTR| (type-token ratio)
4. SEMANTIC_DRIFT      – |real − sim| mean cosine-sim of replies to root tweet
5. EMOTION_TRAJECTORY  – Mean |real − sim| VADER compound per time-bin
6. GINI_DIFF           – |real − sim| Gini coefficient of posts-per-user
7. SPEARMAN_AGGRESSION – Spearman r between per-thread mean aggression (real vs sim)
8. STANCE_CONSISTENCY  – Simulated % of repeat-posting users maintaining
                         same sentiment polarity (higher = better persona fidelity)

Composite score: all metrics normalised 0-1, mean (higher = better).
"""

import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/Users/luketervit/Desktop/Dissertation_Final")
SWEEP_DIR = BASE / "parameter_sweep"
REAL_DIR  = BASE / "batch_simulations_reconstructed"
OUT_DIR   = BASE / "parameter_sweep_analysis"

THREAD_IDS = [
    "thread_001", "thread_005", "thread_010", "thread_015", "thread_020",
    "thread_030", "thread_050", "thread_060", "thread_075", "thread_090",
]

# ── Lazy-load heavy models once ───────────────────────────────────────────────
_vader = None
_embedder = None


def vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("  [model] Loading all-MiniLM-L6-v2 (once) …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Low-level helpers ─────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def reply_events(events: list) -> list:
    """Return non-root temporal events."""
    return [e for e in events if not e.get("is_root", False)]


def texts_from_events(events: list) -> list[str]:
    return [e["text"] for e in reply_events(events)]


# ── Metric helpers ────────────────────────────────────────────────────────────

def sentiment_bucket(texts: list[str]) -> np.ndarray:
    """Return [pos_frac, neg_frac, neu_frac] from VADER compound."""
    va = vader()
    pos = neg = neu = 0
    for t in texts:
        c = va.polarity_scores(t)["compound"]
        if c >= 0.05:
            pos += 1
        elif c <= -0.05:
            neg += 1
        else:
            neu += 1
    total = max(pos + neg + neu, 1)
    return np.array([pos / total, neg / total, neu / total])


def safe_jsd(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def type_token_ratio(texts: list[str]) -> float:
    tokens = []
    for t in texts:
        tokens.extend(re.findall(r"\b[a-z]{2,}\b", t.lower()))
    return len(set(tokens)) / max(len(tokens), 1)


def gini_coeff(counts: list[int]) -> float:
    arr = np.sort(np.array(counts, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def _power_law(x, a, b):
    return a * np.power(np.array(x, dtype=float) + 1, b)


def fit_decay_exponent(epoch_counter: Counter) -> float:
    """Fit y = a*(x+1)^b to (epoch, count) data; return b."""
    if len(epoch_counter) < 3:
        return np.nan
    xs = sorted(epoch_counter)
    ys = [float(epoch_counter[x]) for x in xs]
    try:
        popt, _ = curve_fit(
            _power_law, xs, ys, p0=[max(ys), -0.5], maxfev=8000,
            bounds=([-np.inf, -5], [np.inf, 5]),
        )
        return float(popt[1])
    except Exception:
        return np.nan


def epoch_counter(events: list) -> Counter:
    c: Counter = Counter()
    for e in reply_events(events):
        c[int(e.get("epoch", 0))] += 1
    return c


def vader_compound_per_epoch(events: list) -> dict[int, list[float]]:
    va = vader()
    result: dict[int, list[float]] = defaultdict(list)
    for e in reply_events(events):
        c = va.polarity_scores(e["text"])["compound"]
        result[int(e.get("epoch", 0))].append(c)
    return result


def semantic_sim_per_epoch(
    events: list, root_emb: np.ndarray, emb_model
) -> dict[int, float]:
    """Mean cosine-similarity to root tweet per epoch."""
    from sklearn.metrics.pairwise import cosine_similarity

    epoch_texts: dict[int, list[str]] = defaultdict(list)
    for e in reply_events(events):
        epoch_texts[int(e.get("epoch", 0))].append(e["text"])

    result: dict[int, float] = {}
    for ep, txts in epoch_texts.items():
        embs = emb_model.encode(txts, show_progress_bar=False)
        sims = cosine_similarity(embs, root_emb.reshape(1, -1)).flatten()
        result[ep] = float(np.mean(sims))
    return result


def stance_consistency_sim(history: list) -> float:
    """
    Simulated stance consistency: fraction of repeat-posting users
    whose VADER compound polarity is identical across all their posts.
    """
    va = vader()
    user_pol: dict = defaultdict(list)
    for post in history:
        if post.get("depth", 0) == 0:
            continue  # skip root
        uid = post["user_id"]
        c = va.polarity_scores(post["text"])["compound"]
        pol = 1 if c >= 0.05 else (-1 if c <= -0.05 else 0)
        user_pol[uid].append(pol)

    repeat = {u: v for u, v in user_pol.items() if len(v) >= 2}
    if not repeat:
        return np.nan
    consistent = sum(1 for v in repeat.values() if len(set(v)) == 1)
    return consistent / len(repeat)


# ── Pre-cache real thread data ────────────────────────────────────────────────

def preload_real_threads() -> dict:
    """
    Load real thread metadata + agents once, cache embeddings of all texts.
    Returns dict keyed by thread_id.
    """
    print("\n[cache] Pre-loading real threads and computing embeddings …")
    emb = embedder()
    cache: dict = {}

    for tid in THREAD_IDS:
        meta_path   = REAL_DIR / tid / "thread_metadata.json"
        agents_path = REAL_DIR / tid / "agents_for_thread.csv"

        if not meta_path.exists():
            print(f"  WARNING: {meta_path} not found — skipping {tid}")
            continue

        meta   = load_json(meta_path)
        events = meta.get("temporal_events", [])
        root   = meta["root_tweet"]["text"]
        txts   = texts_from_events(events)

        # Embeddings
        all_txts = [root] + txts
        all_embs = emb.encode(all_txts, show_progress_bar=False)
        root_emb  = all_embs[0]
        reply_embs = all_embs[1:]

        # Agents aggression (real)
        real_agg = np.nan
        if agents_path.exists():
            agents_df = pd.read_csv(agents_path)
            if {"hate_score", "offensive_score"}.issubset(agents_df.columns):
                real_agg = float(
                    (agents_df["hate_score"] + agents_df["offensive_score"]).mean()
                )

        cache[tid] = {
            "events":     events,
            "root_text":  root,
            "root_emb":   root_emb,
            "reply_embs": reply_embs,
            "texts":      txts,
            "real_agg":   real_agg,
        }
        print(f"  [cache] {tid}: {len(txts)} replies, real_agg={real_agg:.3f}")

    return cache


# ── Per-condition analysis ────────────────────────────────────────────────────

def analyse_condition(condition: str, real_cache: dict) -> dict | None:
    """Run all 8 metrics for one condition, pooled across threads."""
    cond_dir = SWEEP_DIR / condition
    emb = embedder()

    # Accumulators
    real_texts_all:      list[str]   = []
    sim_texts_all:       list[str]   = []
    real_scores_bucket:  list        = []
    sim_scores_bucket:   list        = []
    real_epoch_c:        Counter     = Counter()
    sim_epoch_c:         Counter     = Counter()
    real_emot:           dict        = defaultdict(list)
    sim_emot:            dict        = defaultdict(list)
    real_drift:          list        = []
    sim_drift:           list        = []
    real_user_c:         Counter     = Counter()
    sim_user_c:          Counter     = Counter()

    per_thread_real_agg: list[float] = []
    per_thread_sim_agg:  list[float] = []

    sim_history_all:     list        = []
    threads_processed    = 0

    for tid in THREAD_IDS:
        if tid not in real_cache:
            continue

        sim_meta_path = cond_dir / tid / "simulation_output" / "simulated_thread_metadata.json"
        hist_path     = cond_dir / tid / "simulation_output" / "thread_history.json"

        if not sim_meta_path.exists():
            print(f"    WARNING: {condition}/{tid} — sim metadata missing, skipping")
            continue

        try:
            sim_meta   = load_json(sim_meta_path)
            rc         = real_cache[tid]
            real_evts  = rc["events"]
            sim_evts   = sim_meta.get("temporal_events", [])
            root_emb   = rc["root_emb"]

            real_txts  = rc["texts"]
            sim_txts   = texts_from_events(sim_evts)

            real_texts_all.extend(real_txts)
            sim_texts_all.extend(sim_txts)

            # VADER buckets (accumulate raw scores for bucket aggregation later)
            va = vader()
            real_scores_bucket.extend(
                [va.polarity_scores(t)["compound"] for t in real_txts]
            )
            sim_scores_bucket.extend(
                [va.polarity_scores(t)["compound"] for t in sim_txts]
            )

            # Temporal decay
            real_epoch_c += epoch_counter(real_evts)
            sim_epoch_c  += epoch_counter(sim_evts)

            # Emotion trajectory
            for ep, scores in vader_compound_per_epoch(real_evts).items():
                real_emot[ep].extend(scores)
            for ep, scores in vader_compound_per_epoch(sim_evts).items():
                sim_emot[ep].extend(scores)

            # Semantic drift (use pre-computed real embeddings for speed)
            from sklearn.metrics.pairwise import cosine_similarity as cossim
            if len(rc["reply_embs"]) > 0:
                sims = cossim(rc["reply_embs"], root_emb.reshape(1, -1)).flatten()
                real_drift.append(float(np.mean(sims)))
            if sim_txts:
                s_embs = emb.encode(sim_txts, show_progress_bar=False)
                sims_s = cossim(s_embs, root_emb.reshape(1, -1)).flatten()
                sim_drift.append(float(np.mean(sims_s)))

            # Gini: user post counts (real users all "unknown" — use index proxy)
            for i, e in enumerate(reply_events(real_evts)):
                real_user_c[f"real_{i % max(len(real_txts), 1)}"] += 1
            for e in reply_events(sim_evts):
                sim_user_c[str(e.get("user_id", "unk"))] += 1

            # Per-thread aggression
            per_thread_real_agg.append(rc["real_agg"])
            if hist_path.exists():
                hist = load_json(hist_path)
                sim_history_all.extend(hist)
                non_root = [p for p in hist if p.get("depth", 0) > 0]
                if non_root:
                    per_thread_sim_agg.append(
                        float(np.mean([p["aggression"] for p in non_root]))
                    )
                else:
                    per_thread_sim_agg.append(np.nan)
            else:
                per_thread_sim_agg.append(np.nan)

            threads_processed += 1

        except Exception as exc:
            import traceback
            print(f"    ERROR {condition}/{tid}: {exc}")
            traceback.print_exc()
            continue

    if threads_processed == 0:
        return None

    # ── Compute metrics ────────────────────────────────────────────────────────

    def _bucket(scores):
        pos = sum(1 for c in scores if c >= 0.05)
        neg = sum(1 for c in scores if c <= -0.05)
        neu = len(scores) - pos - neg
        t = max(pos + neg + neu, 1)
        return np.array([pos / t, neg / t, neu / t])

    real_b = _bucket(real_scores_bucket)
    sim_b  = _bucket(sim_scores_bucket)

    # 1. Sentiment JSD
    sentiment_jsd = safe_jsd(real_b.copy(), sim_b.copy())

    # 2. Temporal decay
    real_exp = fit_decay_exponent(real_epoch_c)
    sim_exp  = fit_decay_exponent(sim_epoch_c)
    temporal_diff = (
        abs(real_exp - sim_exp)
        if not (np.isnan(real_exp) or np.isnan(sim_exp))
        else np.nan
    )

    # 3. Lexical diversity (TTR)
    real_ttr = type_token_ratio(real_texts_all)
    sim_ttr  = type_token_ratio(sim_texts_all)
    ttr_diff = abs(real_ttr - sim_ttr)

    # 4. Semantic drift
    sem_diff = (
        abs(np.mean(real_drift) - np.mean(sim_drift))
        if real_drift and sim_drift
        else np.nan
    )

    # 5. Emotion trajectory: mean |real - sim| at shared epochs
    shared_epochs = sorted(set(real_emot) & set(sim_emot))
    if len(shared_epochs) >= 3:
        r_emot = [np.mean(real_emot[ep]) for ep in shared_epochs]
        s_emot = [np.mean(sim_emot[ep]) for ep in shared_epochs]
        emot_diff = float(np.mean(np.abs(np.array(r_emot) - np.array(s_emot))))
    else:
        emot_diff = np.nan

    # 6. Gini
    real_gini = gini_coeff(list(real_user_c.values()))
    sim_gini  = gini_coeff(list(sim_user_c.values()))
    gini_diff = abs(real_gini - sim_gini)

    # 7. Spearman r (aggression)
    valid = [
        (r, s)
        for r, s in zip(per_thread_real_agg, per_thread_sim_agg)
        if not (np.isnan(r) or np.isnan(s))
    ]
    if len(valid) >= 3:
        spearman_r, spearman_p = spearmanr(
            [v[0] for v in valid], [v[1] for v in valid]
        )
    else:
        spearman_r, spearman_p = np.nan, np.nan

    # 8. Stance consistency (simulated only)
    stance_sim = (
        stance_consistency_sim(sim_history_all) if sim_history_all else np.nan
    )

    return {
        "condition":         condition,
        "threads_processed": threads_processed,
        # Raw values
        "real_pos":          real_b[0],
        "real_neg":          real_b[1],
        "real_neu":          real_b[2],
        "sim_pos":           sim_b[0],
        "sim_neg":           sim_b[1],
        "sim_neu":           sim_b[2],
        "real_ttr":          real_ttr,
        "sim_ttr":           sim_ttr,
        "real_gini":         real_gini,
        "sim_gini":          sim_gini,
        "real_decay_exp":    real_exp,
        "sim_decay_exp":     sim_exp,
        "real_drift_mean":   np.mean(real_drift) if real_drift else np.nan,
        "sim_drift_mean":    np.mean(sim_drift) if sim_drift else np.nan,
        # Metrics (lower diff = better, except spearman_r)
        "sentiment_jsd":        sentiment_jsd,
        "temporal_decay_diff":  temporal_diff,
        "ttr_diff":             ttr_diff,
        "semantic_drift_diff":  sem_diff,
        "emotion_traj_diff":    emot_diff,
        "gini_diff":            gini_diff,
        "spearman_r":           spearman_r,
        "spearman_p":           spearman_p,
        "stance_consistency":   stance_sim,
    }


# ── Figure helpers ────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("deep")
REAL_COL = PALETTE[0]
SIM_COL  = PALETTE[1]


def save_condition_figures(result: dict, out_dir: Path, condition: str) -> None:
    """Save 4 per-condition figures."""
    fig_dir = out_dir / condition
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig A — Sentiment distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["Positive", "Negative", "Neutral"]
    x = np.arange(len(cats))
    w = 0.35
    real_vals = [result["real_pos"], result["real_neg"], result["real_neu"]]
    sim_vals  = [result["sim_pos"],  result["sim_neg"],  result["sim_neu"]]
    bars_r = ax.bar(x - w / 2, [v * 100 for v in real_vals], w,
                    label="Real", color=REAL_COL)
    bars_s = ax.bar(x + w / 2, [v * 100 for v in sim_vals], w,
                    label="Simulated", color=SIM_COL)
    for bar in list(bars_r) + list(bars_s):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"{condition}\nSentiment Distribution  (JSD={result['sentiment_jsd']:.4f})")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.tight_layout()
    fig.savefig(fig_dir / "A_sentiment_distribution.png", dpi=150)
    plt.close(fig)

    # Fig B — TTR & Gini bar comparison
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (metric, r_val, s_val, label) in zip(axes, [
        ("TTR", result["real_ttr"], result["sim_ttr"], "Type-Token Ratio"),
        ("Gini", result["real_gini"], result["sim_gini"], "Gini Coefficient"),
    ]):
        ax.bar(["Real", "Simulated"], [r_val, s_val],
               color=[REAL_COL, SIM_COL])
        ax.set_title(f"{label}\n(diff={abs(r_val - s_val):.4f})")
        ax.set_ylim(0, max(r_val, s_val) * 1.25)
        for i, v in enumerate([r_val, s_val]):
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    fig.suptitle(condition, fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_dir / "B_lexical_and_gini.png", dpi=150)
    plt.close(fig)

    # Fig C — Decay exponents
    fig, ax = plt.subplots(figsize=(5, 4))
    exps = [result["real_decay_exp"], result["sim_decay_exp"]]
    labels = ["Real", "Simulated"]
    colors = [REAL_COL, SIM_COL]
    valid_mask = [not np.isnan(e) for e in exps]
    ax.bar(
        [l for l, v in zip(labels, valid_mask) if v],
        [e for e, v in zip(exps, valid_mask) if v],
        color=[c for c, v in zip(colors, valid_mask) if v],
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Power-law exponent (b)")
    ax.set_title(
        f"{condition}\nTemporal Decay Exponents"
        f"  (diff={result['temporal_decay_diff']:.3f})"
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "C_temporal_decay.png", dpi=150)
    plt.close(fig)

    # Fig D — Semantic drift bar
    fig, ax = plt.subplots(figsize=(5, 4))
    rd = result.get("real_drift_mean", np.nan)
    sd = result.get("sim_drift_mean", np.nan)
    vals = [rd, sd]
    bars = ax.bar(["Real", "Simulated"], vals, color=[REAL_COL, SIM_COL])
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean cosine-similarity to root tweet")
    diff_label = (
        f"{result['semantic_drift_diff']:.4f}"
        if not np.isnan(result["semantic_drift_diff"])
        else "N/A"
    )
    ax.set_title(f"{condition}\nSemantic Drift  (diff={diff_label})")
    plt.tight_layout()
    fig.savefig(fig_dir / "D_semantic_drift.png", dpi=150)
    plt.close(fig)


def save_summary_figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Save global comparison figures."""

    # Sort by composite score
    df_sorted = df.sort_values("composite_score", ascending=False)

    # Fig 1 — Composite leaderboard (horizontal bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [
        "#2ecc71" if s >= 0.7 else "#f39c12" if s >= 0.5 else "#e74c3c"
        for s in df_sorted["composite_score"]
    ]
    bars = ax.barh(
        df_sorted["condition"][::-1],
        df_sorted["composite_score"][::-1],
        color=colors[::-1],
    )
    ax.axvline(df_sorted["composite_score"].median(), color="black",
               linestyle="--", linewidth=1, label="Median")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Composite Score (0–1, higher = better fidelity)")
    ax.set_title("Parameter Sweep — Composite Leaderboard\n"
                 "(green ≥0.70, amber ≥0.50, red <0.50)")
    ax.legend()
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_composite_leaderboard.png", dpi=200)
    plt.close(fig)

    # Fig 2 — Metric heatmap (normalised values)
    normed_cols = [c for c in df.columns if c.endswith("_normed")]
    if normed_cols:
        heat_df = (
            df.set_index("condition")[normed_cols]
            .rename(columns=lambda c: c.replace("_normed", "").replace("_", "\n"))
            .loc[df_sorted["condition"]]
        )
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Normalised Metric Scores per Condition\n"
                     "(1 = best, 0 = worst in each column)")
        plt.tight_layout()
        fig.savefig(out_dir / "fig2_metric_heatmap.png", dpi=200)
        plt.close(fig)

    # Fig 3 — Sentiment JSD ranking
    fig, ax = plt.subplots(figsize=(10, 7))
    df_j = df_sorted[["condition", "sentiment_jsd"]].copy()
    ax.barh(
        df_j["condition"][::-1],
        df_j["sentiment_jsd"][::-1],
        color=PALETTE[2],
    )
    ax.axvline(0.15, color="red", linestyle="--", linewidth=1.2,
               label="JSD=0.15 (good threshold)")
    ax.set_xlabel("Sentiment JSD (lower = better)")
    ax.set_title("Sentiment JSD by Condition")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_sentiment_jsd_ranking.png", dpi=200)
    plt.close(fig)

    # Fig 4 — Spearman r (aggression rank correlation)
    fig, ax = plt.subplots(figsize=(10, 7))
    df_sp = df_sorted[["condition", "spearman_r"]].dropna(subset=["spearman_r"])
    bars = ax.barh(
        df_sp["condition"][::-1],
        df_sp["spearman_r"][::-1],
        color=PALETTE[3],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="green", linestyle="--", linewidth=1,
               label="r=0.5")
    ax.set_xlabel("Spearman r — aggression rank correlation (higher = better)")
    ax.set_title("Cross-Thread Aggression Rank Correlation by Condition")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_aggression_spearman.png", dpi=200)
    plt.close(fig)

    # Fig 5 — TTR & Gini scatter (real vs sim)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (col_r, col_s, title) in zip(axes, [
        ("real_ttr", "sim_ttr", "Type-Token Ratio"),
        ("real_gini", "sim_gini", "Gini Coefficient"),
    ]):
        ax.scatter(df[col_r], df[col_s], alpha=0.8, s=60, color=PALETTE[4])
        lim = [
            min(df[col_r].min(), df[col_s].min()) * 0.95,
            max(df[col_r].max(), df[col_s].max()) * 1.05,
        ]
        ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
        for _, row in df.iterrows():
            ax.annotate(
                row["condition"],
                (row[col_r], row[col_s]),
                fontsize=5,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"Real {title}")
        ax.set_ylabel(f"Simulated {title}")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Lexical Diversity & Inequality — Real vs Simulated (per condition)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_ttr_gini_scatter.png", dpi=200)
    plt.close(fig)

    # Fig 6 — Stance consistency (simulated)
    df_st = df_sorted[["condition", "stance_consistency"]].dropna(
        subset=["stance_consistency"]
    )
    if not df_st.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            df_st["condition"][::-1],
            df_st["stance_consistency"][::-1],
            color=PALETTE[5],
        )
        ax.set_xlabel("Simulated stance consistency (fraction of repeat-posters)")
        ax.set_title("Stance Consistency — Simulated (higher = better persona fidelity)")
        plt.tight_layout()
        fig.savefig(out_dir / "fig6_stance_consistency.png", dpi=200)
        plt.close(fig)

    print(f"  Saved 6 summary figures to {out_dir}")


# ── Normalisation & composite score ──────────────────────────────────────────

DIFF_METRICS = [
    "sentiment_jsd",
    "temporal_decay_diff",
    "ttr_diff",
    "semantic_drift_diff",
    "emotion_traj_diff",
    "gini_diff",
]
HIGH_METRICS = ["spearman_r", "stance_consistency"]


def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise each metric 0-1; compute mean composite score."""
    df = df.copy()
    normed_cols = []

    for m in DIFF_METRICS:
        col = df[m].copy()
        lo, hi = col.min(), col.max()
        key = m + "_normed"
        if hi - lo < 1e-10:
            df[key] = 0.5
        else:
            # Lower diff → higher score
            df[key] = 1 - (col - lo) / (hi - lo)
        normed_cols.append(key)

    for m in HIGH_METRICS:
        col = df[m].copy().fillna(col.min() if not col.dropna().empty else 0)
        lo, hi = col.min(), col.max()
        key = m + "_normed"
        if hi - lo < 1e-10:
            df[key] = 0.5
        else:
            df[key] = (col - lo) / (hi - lo)
        normed_cols.append(key)

    df["composite_score"] = df[normed_cols].mean(axis=1)
    return df


# ── Leaderboard printing ──────────────────────────────────────────────────────

def print_leaderboard(df: pd.DataFrame) -> None:
    df_s = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    rank_cols = [
        "condition", "composite_score",
        "sentiment_jsd", "temporal_decay_diff", "ttr_diff",
        "semantic_drift_diff", "emotion_traj_diff", "gini_diff",
        "spearman_r", "stance_consistency",
    ]
    present = [c for c in rank_cols if c in df_s.columns]

    print("\n" + "=" * 100)
    print("FINAL LEADERBOARD  (higher composite_score = better simulation fidelity)")
    print("=" * 100)
    print(f"{'Rank':<5} ", end="")
    print(f"{'Condition':<28} ", end="")
    print(f"{'Composite':>9} ", end="")
    print(f"{'SentJSD':>8} ", end="")
    print(f"{'DecayΔ':>8} ", end="")
    print(f"{'TTRΔ':>7} ", end="")
    print(f"{'SemΔ':>7} ", end="")
    print(f"{'EmotΔ':>8} ", end="")
    print(f"{'GiniΔ':>7} ", end="")
    print(f"{'SprmnR':>8} ", end="")
    print(f"{'StncCns':>9}")
    print("-" * 100)

    for i, row in df_s.iterrows():
        def fmt(v, decimals=4):
            return f"{v:.{decimals}f}" if not (isinstance(v, float) and np.isnan(v)) else " N/A  "

        print(f"{i+1:<5} ", end="")
        print(f"{row['condition']:<28} ", end="")
        print(f"{fmt(row['composite_score'], 4):>9} ", end="")
        print(f"{fmt(row['sentiment_jsd']):>8} ", end="")
        print(f"{fmt(row.get('temporal_decay_diff', np.nan), 3):>8} ", end="")
        print(f"{fmt(row['ttr_diff'], 4):>7} ", end="")
        print(f"{fmt(row.get('semantic_drift_diff', np.nan), 4):>7} ", end="")
        print(f"{fmt(row.get('emotion_traj_diff', np.nan), 4):>8} ", end="")
        print(f"{fmt(row['gini_diff'], 4):>7} ", end="")
        print(f"{fmt(row.get('spearman_r', np.nan), 4):>8} ", end="")
        print(f"{fmt(row.get('stance_consistency', np.nan), 4):>9}")

    print("=" * 100)

    best = df_s.iloc[0]
    print(f"\n★  BEST CONDITION: '{best['condition']}'")
    print(f"   Composite score: {best['composite_score']:.4f}")
    print(f"\n   Why it wins:")

    # Find per-metric ranks
    for m, label, low_better in [
        ("sentiment_jsd",       "Sentiment JSD",        True),
        ("temporal_decay_diff", "Temporal decay diff",  True),
        ("ttr_diff",            "TTR diff",             True),
        ("semantic_drift_diff", "Semantic drift diff",  True),
        ("emotion_traj_diff",   "Emotion traj diff",    True),
        ("gini_diff",           "Gini diff",            True),
        ("spearman_r",          "Aggression Spearman r",False),
        ("stance_consistency",  "Stance consistency",   False),
    ]:
        if m not in df_s.columns:
            continue
        col = df_s[m].dropna()
        if col.empty:
            continue
        rank_s = (col.rank(ascending=low_better).astype(int))
        rank_of_best = rank_s.iloc[0]
        val = best[m]
        val_str = f"{val:.4f}" if not np.isnan(val) else "N/A"
        print(f"   - {label:<28}: {val_str}  (rank {rank_of_best}/{len(col)})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pre-load all real thread data (once)
    real_cache = preload_real_threads()

    # 2. Discover conditions
    conditions = sorted(
        d.name for d in SWEEP_DIR.iterdir()
        if d.is_dir() and d.name != "analysis"
    )
    print(f"\nFound {len(conditions)} conditions: {conditions}\n")

    # 3. Analyse each condition
    results = []
    for cond in conditions:
        print(f"\n{'─'*60}")
        print(f"  Condition: {cond}")
        res = analyse_condition(cond, real_cache)
        if res is None:
            print(f"  SKIPPED — no valid data for {cond}")
            continue
        print(
            f"  threads={res['threads_processed']}  "
            f"sent_JSD={res['sentiment_jsd']:.4f}  "
            f"ttr_diff={res['ttr_diff']:.4f}  "
            f"spearman_r={res.get('spearman_r', float('nan')):.3f}"
        )
        save_condition_figures(res, OUT_DIR, cond)
        results.append(res)

    if not results:
        print("ERROR: no results — check paths.")
        sys.exit(1)

    # 4. Composite scoring
    df = pd.DataFrame(results)
    df = add_composite(df)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # 5. Save leaderboard CSV
    leaderboard_cols = (
        ["condition", "composite_score", "threads_processed"]
        + DIFF_METRICS
        + HIGH_METRICS
        + [c for c in df.columns if c.endswith("_normed")]
        + [
            "real_pos", "real_neg", "real_neu",
            "sim_pos", "sim_neg", "sim_neu",
            "real_ttr", "sim_ttr", "real_gini", "sim_gini",
            "real_decay_exp", "sim_decay_exp",
            "real_drift_mean", "sim_drift_mean",
            "spearman_p",
        ]
    )
    export_cols = [c for c in leaderboard_cols if c in df.columns]
    df[export_cols].to_csv(OUT_DIR / "leaderboard.csv", index=False)
    print(f"\n[saved] Leaderboard CSV → {OUT_DIR / 'leaderboard.csv'}")

    # 6. Summary figures
    print("\n[figures] Generating summary figures …")
    save_summary_figures(df, OUT_DIR)

    # 7. Print leaderboard
    print_leaderboard(df)

    # 8. Per-metric rankings table
    print("\n── Per-Metric Rankings ───────────────────────────────────────────")
    for m, low_better in [(m, True) for m in DIFF_METRICS] + [
        ("spearman_r", False), ("stance_consistency", False)
    ]:
        if m not in df.columns:
            continue
        col = df[["condition", m]].dropna()
        col = col.sort_values(m, ascending=low_better).reset_index(drop=True)
        best_c = col.iloc[0]["condition"] if len(col) > 0 else "N/A"
        best_v = col.iloc[0][m] if len(col) > 0 else np.nan
        label = "lower better" if low_better else "higher better"
        print(
            f"  {m:<28}  best={best_c:<28} ({best_v:.4f}, {label})"
        )

    print(f"\n[done] All output saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
