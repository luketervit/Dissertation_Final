#!/usr/bin/env python3
"""Recompute real thread depth from reply chains and regenerate Fig 7.

This script updates real-structure columns in an existing per-thread analysis CSV
without rerunning tweet classification. It derives depth from raw tweet reply links
(`in_reply_to_status_id_str`) in data/*.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def norm_id(value: object) -> str | None:
    """Normalize tweet IDs across int/float/scientific-notation/string forms."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass

    s = str(value).strip()
    if not s or s.lower() in {"nan", "<na>", "none", "null"}:
        return None

    try:
        if "e" in s.lower() or s.endswith(".0"):
            return str(int(float(s)))
        if s.isdigit():
            return s
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def load_thread_targets(real_dir: Path, thread_ids: set[str]) -> tuple[dict[str, set[str]], set[str]]:
    """Load tweet ids per thread from reconstructed real metadata."""
    thread_to_ids: dict[str, set[str]] = {}
    conv_hints: set[str] = set()

    for thread_id in sorted(thread_ids):
        meta_path = real_dir / thread_id / "thread_metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        ids: set[str] = set()

        root_id = norm_id(meta.get("root_tweet", {}).get("id"))
        if root_id:
            ids.add(root_id)

        conv_id = norm_id(meta.get("root_tweet", {}).get("conversation_id"))
        if conv_id:
            conv_hints.add(conv_id)

        for event in meta.get("temporal_events", []):
            tid = norm_id(event.get("tweet_id"))
            if tid:
                ids.add(tid)

        if ids:
            thread_to_ids[thread_id] = ids

    return thread_to_ids, conv_hints


def build_reply_lookup(
    data_dir: Path,
    target_ids: set[str],
    conv_hints: set[str],
    chunksize: int = 200_000,
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Build id->parent and id->conversation maps with two streaming passes."""
    id_to_parent: dict[str, str | None] = {}
    id_to_conv: dict[str, str | None] = {}

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Pass 1/2: locating target tweet ids across {len(csv_files)} CSV files...")
    for csv_path in csv_files:
        for chunk in pd.read_csv(
            csv_path,
            usecols=["id_str", "conversationIdStr", "in_reply_to_status_id_str"],
            dtype="string",
            chunksize=chunksize,
            low_memory=False,
        ):
            id_norm = chunk["id_str"].map(norm_id)
            mask = id_norm.isin(target_ids)
            if not mask.any():
                continue

            sub = chunk.loc[mask, ["id_str", "conversationIdStr", "in_reply_to_status_id_str"]].copy()
            sub["id_norm"] = sub["id_str"].map(norm_id)
            sub["conv_norm"] = sub["conversationIdStr"].map(norm_id)
            sub["parent_norm"] = sub["in_reply_to_status_id_str"].map(norm_id)

            for row in sub.itertuples(index=False):
                tid = row.id_norm
                if not tid:
                    continue
                id_to_parent[tid] = row.parent_norm
                id_to_conv[tid] = row.conv_norm

    conv_ids = set(conv_hints)
    conv_ids.update(c for c in id_to_conv.values() if c)

    print(f"Pass 2/2: loading reply maps for {len(conv_ids)} conversations...")
    for csv_path in csv_files:
        for chunk in pd.read_csv(
            csv_path,
            usecols=["id_str", "conversationIdStr", "in_reply_to_status_id_str"],
            dtype="string",
            chunksize=chunksize,
            low_memory=False,
        ):
            conv_norm = chunk["conversationIdStr"].map(norm_id)
            mask = conv_norm.isin(conv_ids)
            if not mask.any():
                continue

            sub = chunk.loc[mask, ["id_str", "conversationIdStr", "in_reply_to_status_id_str"]].copy()
            sub["id_norm"] = sub["id_str"].map(norm_id)
            sub["conv_norm"] = sub["conversationIdStr"].map(norm_id)
            sub["parent_norm"] = sub["in_reply_to_status_id_str"].map(norm_id)

            for row in sub.itertuples(index=False):
                tid = row.id_norm
                if not tid:
                    continue
                id_to_parent[tid] = row.parent_norm
                id_to_conv[tid] = row.conv_norm

    print(f"Reply lookup size: {len(id_to_parent)} tweets")
    return id_to_parent, id_to_conv


def compute_depth(tweet_id: str, id_to_parent: dict[str, str | None]) -> int:
    """Depth = number of parent hops until root/unknown parent."""
    depth = 0
    seen = {tweet_id}
    current = tweet_id

    while True:
        parent = id_to_parent.get(current)
        if not parent or parent == current:
            return depth

        depth += 1
        if parent in seen:
            # Cycle guard: back off one level and stop.
            return max(depth - 1, 0)

        seen.add(parent)
        current = parent


def recompute_real_structure(
    per_thread: pd.DataFrame,
    thread_to_ids: dict[str, set[str]],
    id_to_parent: dict[str, str | None],
) -> pd.DataFrame:
    """Update real depth/subthread columns in per-thread dataframe."""
    pt = per_thread.copy()

    updates = 0
    for idx, row in pt.iterrows():
        thread_id = str(row["thread_id"])
        ids = thread_to_ids.get(thread_id)
        if not ids:
            continue

        depths = [compute_depth(tid, id_to_parent) for tid in ids]
        if not depths:
            continue

        n_posts = len(ids)
        pt.at[idx, "real_total_posts"] = n_posts
        pt.at[idx, "real_max_depth"] = int(max(depths))
        pt.at[idx, "real_mean_depth"] = float(np.mean(depths))
        pt.at[idx, "real_num_subthreads"] = int(sum(1 for d in depths if d == 1))
        updates += 1

    print(f"Updated real structure for {updates} threads")
    return pt


def make_fig7(pt: pd.DataFrame, out_path: Path, title_suffix: str = "New Params") -> None:
    """Generate updated depth comparison scatter plot."""
    x = pt["real_max_depth"].astype(float).values
    y = pt["sim_max_depth"].astype(float).values

    rng = np.random.default_rng(42)
    x_jitter = x + rng.normal(0, 0.05, len(x))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        x_jitter,
        y,
        alpha=0.65,
        s=50,
        edgecolors="black",
        linewidths=0.5,
        color="#4CAF50",
    )

    max_val = max(float(np.max(x)), float(np.max(y))) + 1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="y = x")

    ax.set_xlabel("Real Thread Max Depth (from reply chains)")
    ax.set_ylabel("Simulated Thread Max Depth")
    ax.set_title(f"Thread Depth: Real vs Simulated ({title_suffix})")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.grid(alpha=0.3)

    real_mean, real_std = float(np.mean(x)), float(np.std(x))
    sim_mean, sim_std = float(np.mean(y)), float(np.std(y))
    ax.text(
        0.97,
        0.05,
        (
            f"Real max depth: {real_mean:.2f} ± {real_std:.2f}\n"
            f"Real range: [{int(np.min(x))}, {int(np.max(x))}]\n"
            f"Sim max depth: {sim_mean:.2f} ± {sim_std:.2f}\n"
            f"Sim range: [{int(np.min(y))}, {int(np.max(y))}]"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute real depth + regenerate Fig 7")
    parser.add_argument(
        "--analysis-dir",
        default="analysis/batch_analysis_new",
        help="Directory containing per_thread_results.csv",
    )
    parser.add_argument(
        "--real-dir",
        default="batch_simulations_reconstructed",
        help="Directory containing real thread_metadata.json files",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing raw USC chunk CSVs",
    )
    parser.add_argument(
        "--dissertation-figure",
        default="dissertation_figures/evaluation_100threads/07_depth_comparison.png",
        help="Optional path to also write updated dissertation figure",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    analysis_dir = project_root / args.analysis_dir
    real_dir = project_root / args.real_dir
    data_dir = project_root / args.data_dir
    dissertation_fig = project_root / args.dissertation_figure

    per_thread_path = analysis_dir / "per_thread_results.csv"
    if not per_thread_path.exists():
        raise FileNotFoundError(f"Missing {per_thread_path}")

    pt = pd.read_csv(per_thread_path)
    thread_ids = set(pt["thread_id"].astype(str).tolist())

    thread_to_ids, conv_hints = load_thread_targets(real_dir, thread_ids)
    all_target_ids: set[str] = set()
    for ids in thread_to_ids.values():
        all_target_ids.update(ids)

    print(f"Threads in analysis: {len(thread_ids)}")
    print(f"Target tweet ids: {len(all_target_ids)}")

    id_to_parent, _ = build_reply_lookup(data_dir, all_target_ids, conv_hints)

    old_unique = sorted(pd.unique(pt["real_max_depth"]).tolist())
    pt_new = recompute_real_structure(pt, thread_to_ids, id_to_parent)
    new_dist = pt_new["real_max_depth"].value_counts().sort_index().to_dict()

    backup_path = analysis_dir / "per_thread_results_flat_backup.csv"
    if not backup_path.exists():
        pt.to_csv(backup_path, index=False)
        print(f"Backup saved: {backup_path}")

    pt_new.to_csv(per_thread_path, index=False)
    print(f"Updated: {per_thread_path}")

    fig_path = analysis_dir / "figures" / "fig7_depth_comparison.png"
    make_fig7(pt_new, fig_path)
    print(f"Updated: {fig_path}")

    if str(dissertation_fig).strip():
        make_fig7(pt_new, dissertation_fig)
        print(f"Updated: {dissertation_fig}")

    print("\nDepth summary")
    print(f"  Previous unique real_max_depth values: {old_unique}")
    print(f"  New real_max_depth distribution: {new_dist}")
    print(f"  New real_max_depth mean: {pt_new['real_max_depth'].mean():.3f}")


if __name__ == "__main__":
    main()
