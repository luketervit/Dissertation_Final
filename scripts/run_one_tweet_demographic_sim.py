#!/usr/bin/env python3
"""Run one root-only simulation with demographic-matched sampled agents.

Workflow:
1. Find a "popular" user with multiple root tweets in a dataset file.
2. Use that user's previous root tweets to collect interacting users.
3. Build demographic strata from those interacting users via agent profiles.
4. Sample a similar synthetic agent population from the profile pool.
5. Simulate ONLY from the most recent root tweet using dolphin-llama3:8b.

This script is intentionally single-run and pragmatic for dissertation checks.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


USER_ID_RE = re.compile(r"'id':\s*(\d+)")


@dataclass
class SelectedUser:
    user_id: int
    root_count: int
    prev_root_ids: list[str]
    target_root_id: str
    target_root_text: str
    target_epoch: float
    popularity_mean_replies: float
    agent_count: int
    interacting_user_ids: set[int]


def norm_id(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "<na>", "none", "null"}:
        return None
    try:
        if "e" in s.lower() or s.endswith(".0"):
            return str(int(float(s)))
        if s.isdigit():
            return s
        # Handle accidental decimal representation.
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def extract_user_id(user_blob: object) -> int | None:
    if user_blob is None or (isinstance(user_blob, float) and np.isnan(user_blob)):
        return None
    m = USER_ID_RE.search(str(user_blob))
    if not m:
        return None
    return int(m.group(1))


def load_tweet_table(path: Path) -> pd.DataFrame:
    cols = [
        "id_str",
        "conversationIdStr",
        "in_reply_to_status_id_str",
        "user",
        "rawContent",
        "text",
        "replyCount",
        "date",
        "epoch",
        "type",
    ]
    df = pd.read_csv(
        path,
        usecols=lambda c: c in cols,
        dtype={
            "id_str": "string",
            "conversationIdStr": "string",
            "in_reply_to_status_id_str": "string",
            "user": "string",
            "rawContent": "string",
            "text": "string",
            "date": "string",
            "type": "string",
        },
    )
    df["id_norm"] = df["id_str"].map(norm_id)
    df["conv_norm"] = df["conversationIdStr"].map(norm_id)
    df["reply_to_norm"] = df["in_reply_to_status_id_str"].map(norm_id)
    df["user_id"] = df["user"].map(extract_user_id)
    df["content"] = df["rawContent"].fillna(df["text"]).fillna("")
    df["reply_count"] = pd.to_numeric(df["replyCount"], errors="coerce").fillna(0)
    df["epoch_num"] = pd.to_numeric(df["epoch"], errors="coerce")
    # Fall back to date ordering where epoch is missing.
    df["date_num"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["user_id"].notna() & df["id_norm"].notna() & df["conv_norm"].notna()].copy()
    df["user_id"] = df["user_id"].astype(int)
    return df


def select_user_and_target_tweet(
    df: pd.DataFrame,
    agent_user_ids: set[int],
    min_root_posts: int,
    min_interactors: int,
    min_agents: int,
    max_agents: int,
    forced_user_id: int | None,
    seed: int,
) -> SelectedUser:
    root_mask = (
        df["reply_to_norm"].isna()
        & df["content"].str.len().ge(20)
        & df["type"].fillna("").str.startswith("tweet")
    )
    roots = df[root_mask].copy()
    roots = roots.sort_values(["epoch_num", "date_num"], na_position="last")

    if forced_user_id is not None:
        user_root = roots[roots["user_id"] == forced_user_id].copy()
        if len(user_root) < min_root_posts:
            raise ValueError(
                f"User {forced_user_id} has only {len(user_root)} root posts; need {min_root_posts}."
            )
        candidate_user_ids = [forced_user_id]
    else:
        root_counts = roots.groupby("user_id").size().reset_index(name="root_count")
        candidate_user_ids = root_counts[root_counts["root_count"] >= min_root_posts]["user_id"].tolist()
        if not candidate_user_ids:
            raise ValueError("No users found with required number of root posts.")

    # Replies inside a root conversation (excluding root tweet itself).
    non_roots = df[(df["id_norm"] != df["conv_norm"]) & df["conv_norm"].notna()].copy()

    rng = random.Random(seed)
    scored: list[SelectedUser] = []
    for uid in candidate_user_ids:
        user_roots = roots[roots["user_id"] == uid].copy()
        user_roots = user_roots.sort_values(["epoch_num", "date_num"], na_position="last")
        if len(user_roots) < min_root_posts:
            continue

        target = user_roots.iloc[-1]
        previous = user_roots.iloc[:-1]
        # Use conversation ids for thread matching because CSV exports may
        # serialize id columns with scientific notation.
        prev_ids = previous["conv_norm"].dropna().astype(str).tolist()
        if len(prev_ids) < (min_root_posts - 1):
            continue

        interactions = non_roots[non_roots["conv_norm"].isin(prev_ids)]
        interacting_user_ids = set(interactions["user_id"].dropna().astype(int).tolist())
        interacting_user_ids.discard(int(uid))

        matched_interactors = interacting_user_ids & agent_user_ids
        if len(matched_interactors) < min_interactors:
            continue

        popularity = float(previous["reply_count"].mean())
        if np.isnan(popularity):
            popularity = 0.0
        agent_count = int(round(popularity))
        if agent_count < min_agents or agent_count > max_agents:
            continue

        scored.append(
            SelectedUser(
                user_id=int(uid),
                root_count=int(len(user_roots)),
                prev_root_ids=prev_ids,
                target_root_id=str(target["id_norm"]),
                target_root_text=str(target["content"]),
                target_epoch=float(target["epoch_num"]) if pd.notna(target["epoch_num"]) else 0.0,
                popularity_mean_replies=popularity,
                agent_count=agent_count,
                interacting_user_ids=matched_interactors,
            )
        )

    if not scored:
        raise ValueError(
            "Could not find a candidate user with enough prior root tweets and matched interacting-user profiles."
        )

    # Highest previous popularity, then most interactors.
    scored.sort(
        key=lambda s: (s.popularity_mean_replies, len(s.interacting_user_ids), s.root_count),
        reverse=True,
    )

    # If forced user id was provided and valid, use first (only) option.
    if forced_user_id is not None:
        return scored[0]

    # Stabilize ties by deterministic random choice among top few.
    top = scored[: min(5, len(scored))]
    return rng.choice(top)


def load_agent_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "user_id",
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
        raise ValueError(f"Agent pool missing required columns: {sorted(missing)}")

    df = df.copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df[df["user_id"].notna()].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["aggression"] = (
        pd.to_numeric(df["hate_score"], errors="coerce").fillna(0)
        + pd.to_numeric(df["offensive_score"], errors="coerce").fillna(0)
    )
    df["aggression_bin"] = pd.cut(
        df["aggression"],
        bins=[-np.inf, 0.30, 0.70, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return df


def sample_demographic_matched_agents(
    pool: pd.DataFrame,
    interacting_user_ids: set[int],
    target_count: int,
    seed: int,
    exclude_user_id: int,
) -> pd.DataFrame:
    interactions = pool[pool["user_id"].isin(interacting_user_ids)].copy()
    if interactions.empty:
        raise ValueError("No interacting users found inside the agent pool.")

    interactions["stratum"] = (
        interactions["political_label"].astype(str) + "|" + interactions["aggression_bin"].astype(str)
    )
    pool = pool.copy()
    pool["stratum"] = pool["political_label"].astype(str) + "|" + pool["aggression_bin"].astype(str)

    candidate_pool = pool[
        (~pool["user_id"].isin(interacting_user_ids)) & (pool["user_id"] != int(exclude_user_id))
    ].copy()
    if candidate_pool.empty:
        raise ValueError("Candidate pool is empty after exclusions.")

    dist = interactions["stratum"].value_counts(normalize=True)

    raw_targets = {k: target_count * v for k, v in dist.items()}
    initial = {k: int(np.floor(v)) for k, v in raw_targets.items()}
    remainder = target_count - sum(initial.values())
    remainders = sorted(
        ((k, raw_targets[k] - initial[k]) for k in raw_targets),
        key=lambda t: t[1],
        reverse=True,
    )
    for i in range(remainder):
        initial[remainders[i % len(remainders)][0]] += 1

    rng = np.random.default_rng(seed)
    selected_chunks: list[pd.DataFrame] = []
    taken_ids: set[int] = set()

    for stratum, n_wanted in initial.items():
        if n_wanted <= 0:
            continue
        candidates = candidate_pool[
            (candidate_pool["stratum"] == stratum)
            & (~candidate_pool["user_id"].isin(taken_ids))
        ]
        if candidates.empty:
            continue
        n_take = min(n_wanted, len(candidates))
        idx = rng.choice(candidates.index.to_numpy(), size=n_take, replace=False)
        chunk = candidates.loc[idx]
        selected_chunks.append(chunk)
        taken_ids.update(chunk["user_id"].tolist())

    selected = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame(columns=pool.columns)

    # Backfill if any strata were under-filled.
    missing = target_count - len(selected)
    if missing > 0:
        fallback = candidate_pool[~candidate_pool["user_id"].isin(taken_ids)]
        if fallback.empty:
            raise ValueError("No fallback agents available to satisfy requested agent count.")
        n_take = min(missing, len(fallback))
        idx = rng.choice(fallback.index.to_numpy(), size=n_take, replace=False)
        selected = pd.concat([selected, fallback.loc[idx]], ignore_index=True)

    if len(selected) < target_count:
        # Final fallback: sample with replacement from selected itself.
        extra_needed = target_count - len(selected)
        if selected.empty:
            raise ValueError("Selected agent set is empty; cannot upsample.")
        idx = rng.choice(selected.index.to_numpy(), size=extra_needed, replace=True)
        extra = selected.loc[idx].copy()
        # Ensure unique user IDs for Mesa by offsetting synthetic IDs.
        max_uid = int(pool["user_id"].max()) + 1
        extra = extra.reset_index(drop=True)
        extra["user_id"] = [max_uid + i for i in range(len(extra))]
        selected = pd.concat([selected, extra], ignore_index=True)

    selected = selected.head(target_count).copy()
    # Columns expected by ThreadModel.
    keep = [
        "user_id",
        "political_label",
        "political_score",
        "emotion_label",
        "emotion_score",
        "sentiment_label",
        "sentiment_score",
        "hate_score",
        "offensive_score",
    ]
    return selected[keep]


def write_thread_inputs(
    run_dir: Path,
    selected: SelectedUser,
    agents_df: pd.DataFrame,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_rounds: int,
) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    agents_path = run_dir / "agents_for_thread.csv"
    metadata_path = run_dir / "thread_metadata.json"
    config_path = run_dir / "config.yaml"

    agents_df.to_csv(agents_path, index=False)

    metadata = {
        "root_tweet": {
            "id": selected.target_root_id,
            "conversation_id": selected.target_root_id,
            "user_id": selected.user_id,
            "text": selected.target_root_text,
            "timestamp": "target_root_only",
            "epoch": selected.target_epoch,
            "expected_replies": selected.agent_count,
            "view_count": 0,
            "like_count": 0,
            "retweet_count": 0,
        },
        "temporal_events": [
            {
                "tweet_id": selected.target_root_id,
                "user_id": selected.user_id,
                "timestamp": "target_root_only",
                "epoch": selected.target_epoch,
                "seconds_since_start": 0,
                "is_root": True,
                "text": selected.target_root_text,
            }
        ],
        "source": "demographic_matched_single_tweet",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "target_tweet_id": selected.target_root_id,
        "paths": {
            "thread_metadata": str(metadata_path.resolve()),
            "agents_for_thread": str(agents_path.resolve()),
        },
        "abm": {
            "lurker_ratio": 0,
            "lurker_dist_strategy": "match_active_agents",
            "bounded_confidence_threshold": 0.3,
            "backfire_threshold": 0.6,
            "backfire_aggression_min": 0.7,
        },
        "llm": {
            "provider": "ollama",
            "model": model_name,
            "api_key_env": None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "simulation": {
            "max_rounds": max_rounds,
            "thread_num": 1,
            "use_few_shot": False,
        },
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return agents_path, metadata_path, config_path


def run_simulation(config_path: Path, out_dir: Path, max_rounds: int) -> None:
    from sim.thread_simulation import ThreadModel

    model = ThreadModel(config_path=str(config_path))
    model.run(max_rounds=max_rounds)
    model.export_results(output_dir=str((out_dir / "simulation_output").resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="One-tweet demographic-matched simulation")
    parser.add_argument("--data-file", default="data/may_july_chunk_1.csv")
    parser.add_argument("--agent-pool", default="processed_agents/processed_agents_chunk_1.csv")
    parser.add_argument("--output-dir", default="output/demographic_one_tweet")
    parser.add_argument("--user-id", type=int, default=None, help="Optional forced user id")
    parser.add_argument("--min-root-posts", type=int, default=2)
    parser.add_argument("--min-interactors", type=int, default=8)
    parser.add_argument("--min-agents", type=int, default=10)
    parser.add_argument("--max-agents", type=int, default=200)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model", default="dolphin-llama3:8b")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_file)
    agent_pool_path = Path(args.agent_pool)
    output_dir = Path(args.output_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not agent_pool_path.exists():
        raise FileNotFoundError(f"Agent pool file not found: {agent_pool_path}")

    print(f"Loading tweets from {data_path} ...")
    tweets = load_tweet_table(data_path)
    print(f"Loaded {len(tweets):,} tweets with usable ids/users")

    print(f"Loading agent pool from {agent_pool_path} ...")
    pool = load_agent_pool(agent_pool_path)
    print(f"Loaded {len(pool):,} agent profiles")

    selected = select_user_and_target_tweet(
        df=tweets,
        agent_user_ids=set(pool["user_id"].tolist()),
        min_root_posts=args.min_root_posts,
        min_interactors=args.min_interactors,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        forced_user_id=args.user_id,
        seed=args.seed,
    )

    print("\nSelected target setup:")
    print(f"  user_id: {selected.user_id}")
    print(f"  root_posts: {selected.root_count}")
    print(f"  previous roots used for demographics: {len(selected.prev_root_ids)}")
    print(f"  previous mean replies (popularity): {selected.popularity_mean_replies:.2f}")
    print(f"  interacting users matched in pool: {len(selected.interacting_user_ids)}")
    print(f"  sampled agent count: {selected.agent_count}")
    print(f"  target root id: {selected.target_root_id}")

    sampled_agents = sample_demographic_matched_agents(
        pool=pool,
        interacting_user_ids=selected.interacting_user_ids,
        target_count=selected.agent_count,
        seed=args.seed,
        exclude_user_id=selected.user_id,
    )

    run_name = f"user_{selected.user_id}_root_{selected.target_root_id}"
    run_dir = output_dir / run_name
    agents_path, metadata_path, config_path = write_thread_inputs(
        run_dir=run_dir,
        selected=selected,
        agents_df=sampled_agents,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
    )

    summary = {
        "user_id": selected.user_id,
        "root_count": selected.root_count,
        "previous_root_ids": selected.prev_root_ids,
        "target_root_id": selected.target_root_id,
        "target_root_text": selected.target_root_text,
        "mean_previous_reply_count": selected.popularity_mean_replies,
        "agent_count": selected.agent_count,
        "matched_interactors": len(selected.interacting_user_ids),
        "agent_pool_file": str(agent_pool_path),
        "data_file": str(data_path),
        "agents_csv": str(agents_path),
        "thread_metadata_json": str(metadata_path),
        "config_yaml": str(config_path),
        "model": args.model,
        "use_few_shot": False,
        "root_only": True,
    }
    with open(run_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPrepared inputs in: {run_dir}")
    if args.dry_run:
        print("Dry run enabled; simulation not executed.")
        return

    print("\nRunning simulation with local Ollama model...")
    run_simulation(config_path=config_path, out_dir=run_dir, max_rounds=args.max_rounds)
    print(f"Simulation completed. Outputs in: {run_dir / 'simulation_output'}")


if __name__ == "__main__":
    main()
