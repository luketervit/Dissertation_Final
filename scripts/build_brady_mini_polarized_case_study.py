"""
Build a mini Brady case-study pack with polarized agents from 5-model outputs.

Design:
- 1 base tweet family per topic (climate, gun, marriage)
- 4 conditions per family: control + edited_1 + edited_2 + edited_3
- Same 100 sampled agents used across all threads
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    txt = str(value)
    return "" if txt.lower() == "nan" else txt


def stable_mode(series: pd.Series, default: str) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return default
    return values.value_counts().idxmax()


def load_llm_defaults(project_root: Path) -> dict:
    cfg_path = project_root / "config" / "thread_config.yaml"
    if not cfg_path.exists():
        return {
            "provider": "ollama",
            "model": "dolphin-llama3:8b",
            "api_key_env": None,
            "temperature": 0.9,
            "max_tokens": 150,
        }
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = cfg.get("llm", {})
    llm.setdefault("provider", "ollama")
    llm.setdefault("model", "dolphin-llama3:8b")
    llm.setdefault("api_key_env", None)
    llm.setdefault("temperature", 0.9)
    llm.setdefault("max_tokens", 150)
    return llm


def load_agent_pool(pattern: str) -> pd.DataFrame:
    files = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No agent files found for glob: {pattern}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        required = {
            "user_id",
            "tweet_count",
            "view_count",
            "reply_count",
            "political_label",
            "political_score",
            "emotion_label",
            "emotion_score",
            "sentiment_label",
            "sentiment_score",
            "hate_score",
            "offensive_score",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fp} missing columns: {sorted(missing)}")
        frames.append(df[list(required)].copy())

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.dropna(subset=["user_id"]).copy()
    pool["user_id"] = pool["user_id"].astype(str)

    for col in [
        "tweet_count",
        "view_count",
        "reply_count",
        "political_score",
        "emotion_score",
        "sentiment_score",
        "hate_score",
        "offensive_score",
    ]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce")

    pool = pool.dropna(
        subset=[
            "political_label",
            "emotion_label",
            "sentiment_label",
            "political_score",
            "emotion_score",
            "sentiment_score",
            "hate_score",
            "offensive_score",
        ]
    ).copy()

    # Aggregate duplicates across chunks.
    agg = (
        pool.groupby("user_id", as_index=False)
        .agg(
            tweet_count=("tweet_count", "sum"),
            view_count=("view_count", "sum"),
            reply_count=("reply_count", "sum"),
            political_score=("political_score", "mean"),
            emotion_score=("emotion_score", "mean"),
            sentiment_score=("sentiment_score", "mean"),
            hate_score=("hate_score", "mean"),
            offensive_score=("offensive_score", "mean"),
            political_label=("political_label", lambda s: stable_mode(s, "Center")),
            emotion_label=("emotion_label", lambda s: stable_mode(s, "optimism")),
            sentiment_label=("sentiment_label", lambda s: stable_mode(s, "neutral")),
        )
        .copy()
    )
    agg["username"] = agg["user_id"]
    agg["role"] = "active"
    agg["aggression"] = agg["hate_score"] + agg["offensive_score"]
    return agg


def _sample_without_replacement(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    n = min(n, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]].copy()


def sample_polarized_agents(pool: pd.DataFrame, n_agents: int, seed: int) -> pd.DataFrame:
    pool = pool.copy()
    pool["political_label"] = pool["political_label"].astype(str).str.title()
    pool = pool[pool["political_label"].isin(["Left", "Center", "Right"])].copy()
    if len(pool) < n_agents:
        raise ValueError(f"Not enough agents in pool ({len(pool)}) for n_agents={n_agents}")

    # Aggression tiers encourage personality diversity.
    pool["aggression_bin"] = pd.qcut(
        pool["aggression"],
        q=3,
        labels=["low", "mid", "high"],
        duplicates="drop",
    )
    if pool["aggression_bin"].isna().all():
        pool["aggression_bin"] = "mid"
    else:
        pool["aggression_bin"] = pool["aggression_bin"].astype(str).replace("nan", "mid")

    rng = random.Random(seed)
    targets = {"Left": 40, "Right": 40, "Center": 20}
    if n_agents != 100:
        # Keep the same polarization proportions for arbitrary sizes.
        left = int(round(0.4 * n_agents))
        right = int(round(0.4 * n_agents))
        center = n_agents - left - right
        targets = {"Left": left, "Right": right, "Center": center}

    sampled_parts = []
    used_ids: set[str] = set()
    bin_weights = {"high": 0.4, "low": 0.4, "mid": 0.2}

    for label, target in targets.items():
        sub = pool[pool["political_label"] == label].copy()
        if sub.empty:
            continue

        label_take = []
        for bin_name in ["high", "low", "mid"]:
            want = int(round(target * bin_weights[bin_name]))
            bin_df = sub[
                (sub["aggression_bin"] == bin_name) & (~sub["user_id"].isin(used_ids))
            ]
            pick = _sample_without_replacement(bin_df, want, rng)
            if not pick.empty:
                used_ids.update(pick["user_id"].tolist())
                label_take.append(pick)

        taken = sum(len(x) for x in label_take)
        if taken < target:
            remaining = sub[~sub["user_id"].isin(used_ids)]
            pick = _sample_without_replacement(remaining, target - taken, rng)
            if not pick.empty:
                used_ids.update(pick["user_id"].tolist())
                label_take.append(pick)

        if label_take:
            sampled_parts.append(pd.concat(label_take, ignore_index=True))

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()

    if len(sampled) < n_agents:
        remaining = pool[~pool["user_id"].isin(sampled["user_id"])]
        # Fill with politically polarized users first.
        prioritized = remaining[remaining["political_label"].isin(["Left", "Right"])]
        fill = _sample_without_replacement(prioritized, n_agents - len(sampled), rng)
        sampled = pd.concat([sampled, fill], ignore_index=True)
        if len(sampled) < n_agents:
            remaining = pool[~pool["user_id"].isin(sampled["user_id"])]
            fill2 = _sample_without_replacement(remaining, n_agents - len(sampled), rng)
            sampled = pd.concat([sampled, fill2], ignore_index=True)

    sampled = sampled.head(n_agents).copy()
    sampled = sampled[
        [
            "user_id",
            "username",
            "political_label",
            "political_score",
            "emotion_label",
            "emotion_score",
            "sentiment_label",
            "sentiment_score",
            "hate_score",
            "offensive_score",
            "role",
            "tweet_count",
            "view_count",
            "reply_count",
        ]
    ].reset_index(drop=True)
    return sampled


def select_families(family_manifest: pd.DataFrame) -> list[str]:
    chosen = []
    for topic in ["climate", "gun", "marriage"]:
        topic_df = family_manifest[family_manifest["topic"] == topic].copy()
        if topic_df.empty:
            raise ValueError(f"No families available for topic '{topic}'")
        # Prefer rows with swap operations because they are stronger counterfactuals.
        swap = topic_df[topic_df["operation"].isin(["swap_neg_to_pos", "swap_pos_to_neg"])]
        family_id = (
            sorted(swap["family_id"].unique())[0]
            if not swap.empty
            else sorted(topic_df["family_id"].unique())[0]
        )
        chosen.append(family_id)
    return chosen


def load_control_metadata(source_thread_manifest: pd.DataFrame, family_id: str) -> dict:
    row = source_thread_manifest[
        (source_thread_manifest["family_id"] == family_id)
        & (source_thread_manifest["condition"] == "control")
    ]
    if row.empty:
        raise ValueError(f"Missing control thread for family {family_id}")
    thread_dir = Path(row.iloc[0]["thread_dir"])
    meta_path = thread_dir / "thread_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_config(thread_dir: Path, tweet_id: str, llm_cfg: dict, thread_num: int, max_rounds: int) -> None:
    config = {
        "target_tweet_id": tweet_id,
        "paths": {
            "thread_metadata": "thread_metadata.json",
            "agents_for_thread": "agents_for_thread.csv",
        },
        "abm": {
            "lurker_ratio": 0.0,
            "lurker_dist_strategy": "match_active_agents",
            "bounded_confidence_threshold": 0.3,
            "backfire_threshold": 0.6,
            "backfire_aggression_min": 0.7,
        },
        "llm": llm_cfg,
        "simulation": {
            "thread_num": thread_num,
            "max_rounds": max_rounds,
            "use_few_shot": False,
        },
    }
    with open(thread_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def build_case_study(
    project_root: Path,
    source_case_root: Path,
    output_root: Path,
    agent_glob: str,
    n_agents: int,
    max_rounds: int,
    seed: int,
    overwrite: bool,
) -> None:
    if output_root.exists() and overwrite:
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    threads_root = output_root / "threads"
    manifest_root = output_root / "manifests"
    threads_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    family_manifest_path = source_case_root / "manifests" / "family_manifest.csv"
    thread_manifest_path = source_case_root / "manifests" / "thread_manifest.csv"
    if not family_manifest_path.exists() or not thread_manifest_path.exists():
        raise FileNotFoundError(
            "Source case study manifests not found. "
            f"Expected: {family_manifest_path} and {thread_manifest_path}"
        )

    family_df = pd.read_csv(family_manifest_path)
    thread_df = pd.read_csv(thread_manifest_path)

    selected_families = select_families(family_df)
    selected_family_df = family_df[family_df["family_id"].isin(selected_families)].copy()

    pool = load_agent_pool(agent_glob)
    agents_df = sample_polarized_agents(pool=pool, n_agents=n_agents, seed=seed)

    llm_cfg = load_llm_defaults(project_root)

    thread_rows = []
    thread_counter = 1
    for family_id in selected_families:
        fam_rows = selected_family_df[selected_family_df["family_id"] == family_id].copy()
        if fam_rows.empty:
            continue
        fam_rows = fam_rows.sort_values("edit_id")
        base = fam_rows.iloc[0]

        control_meta = load_control_metadata(thread_df, family_id)
        root = control_meta["root_tweet"]
        root_tweet_id = str(root["id"])
        root_user_id = str(root["user_id"])
        root_ts = str(root.get("timestamp", ""))
        root_views = int(root.get("view_count", 0) or 0)

        variants = [
            {
                "condition": "control",
                "tweet_text": base["base_text"],
                "operation": "control",
                "lexeme_before": "",
                "lexeme_after": "",
            }
        ]
        for _, r in fam_rows.iterrows():
            variants.append(
                {
                    "condition": r["edit_id"],
                    "tweet_text": r["edited_text"],
                    "operation": r["operation"],
                    "lexeme_before": safe_text(r.get("lexeme_before", "")),
                    "lexeme_after": safe_text(r.get("lexeme_after", "")),
                }
            )

        for variant in variants:
            thread_id = f"thread_{thread_counter:03d}"
            thread_dir = threads_root / thread_id
            thread_dir.mkdir(parents=True, exist_ok=True)

            agents_df.to_csv(thread_dir / "agents_for_thread.csv", index=False)

            tweet_text = str(variant["tweet_text"])
            metadata = {
                "root_tweet": {
                    "id": root_tweet_id,
                    "conversation_id": root_tweet_id,
                    "user_id": root_user_id,
                    "text": tweet_text,
                    "expected_replies": 0,
                    "timestamp": root_ts,
                    "view_count": root_views,
                },
                "replies": {"actual_count": 0, "tweet_ids": []},
                "temporal_events": [
                    {
                        "tweet_id": root_tweet_id,
                        "user_id": root_user_id,
                        "timestamp": root_ts,
                        "epoch": 0,
                        "seconds_since_start": 0,
                        "is_root": True,
                        "text": tweet_text,
                    }
                ],
                "source": "brady_mini_polarized_case_study",
                "case_study": {
                    "family_id": family_id,
                    "condition": variant["condition"],
                    "topic": str(base["topic"]),
                    "operation": variant["operation"],
                    "lexeme_before": variant["lexeme_before"],
                    "lexeme_after": variant["lexeme_after"],
                    "base_tweet_id": str(base["base_tweet_id"]),
                    "base_user_id": str(base["base_user_id"]),
                },
            }
            with open(thread_dir / "thread_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            write_config(
                thread_dir=thread_dir,
                tweet_id=root_tweet_id,
                llm_cfg=llm_cfg,
                thread_num=thread_counter,
                max_rounds=max_rounds,
            )

            thread_rows.append(
                {
                    "thread_id": thread_id,
                    "thread_num": thread_counter,
                    "family_id": family_id,
                    "condition": variant["condition"],
                    "topic": str(base["topic"]),
                    "operation": variant["operation"],
                    "lexeme_before": variant["lexeme_before"],
                    "lexeme_after": variant["lexeme_after"],
                    "base_tweet_id": str(base["base_tweet_id"]),
                    "base_user_id": str(base["base_user_id"]),
                    "tweet_text": tweet_text,
                    "agents_count": int(len(agents_df)),
                    "thread_dir": str(thread_dir),
                }
            )
            thread_counter += 1

    thread_manifest_out = pd.DataFrame(thread_rows)
    if thread_manifest_out.empty:
        raise ValueError("No thread packages were created.")
    thread_manifest_out.to_csv(manifest_root / "thread_manifest.csv", index=False)
    selected_family_df.to_csv(manifest_root / "family_manifest.csv", index=False)

    # Agent diagnostics for transparency.
    agent_diag = {
        "n_agents": int(len(agents_df)),
        "political_dist": {
            k: int(v) for k, v in agents_df["political_label"].value_counts().to_dict().items()
        },
        "emotion_dist": {
            k: int(v) for k, v in agents_df["emotion_label"].value_counts().to_dict().items()
        },
        "mean_hate_score": float(pd.to_numeric(agents_df["hate_score"]).mean()),
        "mean_offensive_score": float(pd.to_numeric(agents_df["offensive_score"]).mean()),
        "mean_aggression": float(
            (pd.to_numeric(agents_df["hate_score"]) + pd.to_numeric(agents_df["offensive_score"])).mean()
        ),
    }
    with open(manifest_root / "agent_summary.json", "w", encoding="utf-8") as f:
        json.dump(agent_diag, f, indent=2)
    agents_df.to_csv(manifest_root / "sampled_agents.csv", index=False)

    total_threads = int(thread_manifest_out["thread_num"].max())
    rel_threads = output_root.relative_to(project_root) / "threads"
    rel_results = output_root.relative_to(project_root) / "results"
    run_script = f"""#!/usr/bin/env bash
set -euo pipefail

python scripts/run_model_comparison_batch.py \\
  --model llama \\
  --input {rel_threads} \\
  --reference '' \\
  --expected-threads {total_threads} \\
  --start 1 --end {total_threads} \\
  --output-root {rel_results} \\
  --max-rounds {max_rounds}
"""
    run_path = manifest_root / "run_commands.sh"
    run_path.write_text(run_script, encoding="utf-8")
    run_path.chmod(0o755)

    summary = {
        "selected_families": selected_families,
        "topics": sorted(thread_manifest_out["topic"].unique().tolist()),
        "threads": int(len(thread_manifest_out)),
        "conditions": sorted(thread_manifest_out["condition"].unique().tolist()),
        "agents_count": int(len(agents_df)),
        "output_root": str(output_root),
        "run_script": str(run_path),
    }
    with open(manifest_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Mini polarized case-study prepared")
    print(f"  Families: {len(selected_families)}")
    print(f"  Threads:  {len(thread_manifest_out)}")
    print(f"  Agents:   {len(agents_df)}")
    print(f"  Output:   {output_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mini Brady case-study with polarized agents.")
    p.add_argument(
        "--source-case-root",
        type=Path,
        default=Path("case_study/brady_mec"),
        help="Existing Brady case-study root with manifests.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("case_study/brady_mini_polarized"),
        help="Output directory for mini case-study.",
    )
    p.add_argument(
        "--agent-glob",
        type=str,
        default="processed_agents/processed_agents_chunk_*.csv",
        help="Glob for 5-model processed agent CSVs.",
    )
    p.add_argument("--agents", type=int, default=100, help="Number of sampled agents.")
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    source_case_root = (project_root / args.source_case_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    build_case_study(
        project_root=project_root,
        source_case_root=source_case_root,
        output_root=output_root,
        agent_glob=args.agent_glob,
        n_agents=args.agents,
        max_rounds=args.max_rounds,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
