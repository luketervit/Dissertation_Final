#!/usr/bin/env python3
"""
Forward-prediction pipeline for one account with strict holdout.

Phases:
1) personalities
   - Find target account root tweets in chronological order
   - Use first N roots as training, last M roots as holdout
   - Build responder personality pool from training-root conversations
   - Sample ~K synthetic agents from responder personalities
   - Materialize 0-shot simulation inputs for each holdout root

2) simulate
   - Run ThreadModel for each prepared holdout run

3) analyze
   - Compare simulated vs real holdout outcomes (sentiment-focused)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

USER_ID_RE = re.compile(r"'id':\s*(\d+)")
USERNAME_RE = re.compile(r"'username':\s*'([^']+)'")

AGENT_KEEP_COLS = [
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


def norm_id(value: object) -> str | None:
    """Normalize raw ID fields into stable string IDs."""
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
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def pick_column(df: pd.DataFrame, names: list[str]) -> pd.Series:
    """Return the first available column from names, else all-None series."""
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([None] * len(df), index=df.index)


def read_usecols(path: Path, wanted: list[str]) -> list[str]:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in wanted if c in cols]


def stable_mode(series: pd.Series) -> object:
    series = series.dropna()
    if len(series) == 0:
        return None
    mode = series.mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return series.iloc[0]


def find_target_root_tweets(
    data_files: list[Path],
    target_user_id: str,
) -> tuple[pd.DataFrame, str]:
    """Extract target user's tweets and identify root tweets."""
    rows: list[pd.DataFrame] = []
    usernames: Counter[str] = Counter()

    wanted = [
        "id",
        "id_str",
        "conversationId",
        "conversationIdStr",
        "inReplyToTweetId",
        "in_reply_to_status_id_str",
        "replyCount",
        "rawContent",
        "text",
        "date",
        "epoch",
        "user",
    ]

    for path in data_files:
        usecols = read_usecols(path, wanted)
        if "user" not in usecols:
            continue

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=25000):
            user_blob = chunk["user"].astype(str)
            user_ids = user_blob.str.extract(USER_ID_RE, expand=False)
            match = user_ids == target_user_id
            if not match.any():
                continue

            sub = chunk.loc[match].copy()
            sub_user_blob = sub["user"].astype(str)
            sub_usernames = sub_user_blob.str.extract(
                USERNAME_RE, expand=False
            ).dropna()
            usernames.update(sub_usernames.tolist())

            tweet_id = pick_column(sub, ["id_str", "id"]).map(norm_id)
            conv_id = pick_column(
                sub, ["conversationIdStr", "conversationId"]
            ).map(norm_id)
            in_reply = pick_column(
                sub, ["in_reply_to_status_id_str", "inReplyToTweetId"]
            ).map(norm_id)
            reply_count = pd.to_numeric(
                pick_column(sub, ["replyCount"]), errors="coerce"
            ).fillna(0)
            text = pick_column(sub, ["rawContent", "text"]).fillna("")
            date = pd.to_datetime(
                pick_column(sub, ["date"]), errors="coerce"
            )
            epoch = pd.to_numeric(
                pick_column(sub, ["epoch"]), errors="coerce"
            )

            rows.append(
                pd.DataFrame(
                    {
                        "tweet_id": tweet_id,
                        "conversation_id": conv_id,
                        "in_reply_to": in_reply,
                        "reply_count": reply_count,
                        "text": text.astype(str),
                        "date": date,
                        "epoch": epoch,
                        "source_file": path.name,
                    }
                )
            )

    if not rows:
        raise ValueError("No tweets found for target user ID.")

    tweets = pd.concat(rows, ignore_index=True)
    tweets = tweets[tweets["tweet_id"].notna()].copy()
    tweets = tweets.sort_values(["tweet_id", "epoch", "date"])
    tweets = tweets.drop_duplicates(subset=["tweet_id"], keep="last")

    no_parent = tweets["in_reply_to"].isna()
    same_conv = tweets["tweet_id"] == tweets["conversation_id"]
    tweets["is_root"] = no_parent | same_conv

    roots = tweets[tweets["is_root"]].copy()
    roots = roots.sort_values(["epoch", "date", "tweet_id"], na_position="last")
    roots = roots.reset_index(drop=True)

    if len(usernames) == 0:
        username = "unknown"
    else:
        username = usernames.most_common(1)[0][0]

    return roots, username


def collect_conversation_events(
    data_files: list[Path],
    conversation_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Collect full conversation events for selected conversation IDs."""
    wanted = [
        "id",
        "id_str",
        "conversationId",
        "conversationIdStr",
        "inReplyToTweetId",
        "in_reply_to_status_id_str",
        "rawContent",
        "text",
        "date",
        "epoch",
        "replyCount",
        "user",
    ]
    by_conv: dict[str, list[dict[str, Any]]] = {cid: [] for cid in conversation_ids}

    for path in data_files:
        usecols = read_usecols(path, wanted)
        if "user" not in usecols:
            continue

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=25000):
            conv_id = pick_column(
                chunk, ["conversationIdStr", "conversationId"]
            ).map(norm_id)
            match = conv_id.isin(conversation_ids)
            if not match.any():
                continue

            sub = chunk.loc[match].copy()
            sub_conv = conv_id.loc[sub.index]

            sub_tweet_id = pick_column(sub, ["id_str", "id"]).map(norm_id)
            sub_user_blob = pick_column(sub, ["user"]).astype(str)
            sub_user_id = sub_user_blob.str.extract(USER_ID_RE, expand=False)
            sub_date = pd.to_datetime(
                pick_column(sub, ["date"]), errors="coerce"
            )
            sub_epoch = pd.to_numeric(
                pick_column(sub, ["epoch"]), errors="coerce"
            )
            sub_reply_count = pd.to_numeric(
                pick_column(sub, ["replyCount"]), errors="coerce"
            ).fillna(0)
            sub_text = pick_column(sub, ["rawContent", "text"]).fillna("")

            for idx in sub.index:
                cid = sub_conv.loc[idx]
                if cid is None:
                    continue
                by_conv[cid].append(
                    {
                        "tweet_id": sub_tweet_id.loc[idx],
                        "conversation_id": cid,
                        "user_id": sub_user_id.loc[idx],
                        "text": str(sub_text.loc[idx]),
                        "date": (
                            sub_date.loc[idx].isoformat()
                            if pd.notna(sub_date.loc[idx])
                            else None
                        ),
                        "epoch": (
                            float(sub_epoch.loc[idx])
                            if pd.notna(sub_epoch.loc[idx])
                            else None
                        ),
                        "reply_count": int(sub_reply_count.loc[idx]),
                    }
                )

    normalized: dict[str, list[dict[str, Any]]] = {}
    for cid, events in by_conv.items():
        if not events:
            normalized[cid] = []
            continue
        evt_df = pd.DataFrame(events)
        evt_df = evt_df.sort_values(
            ["epoch", "date", "tweet_id"], na_position="last"
        )
        records: list[dict[str, Any]] = []
        root_found = False
        start_epoch = evt_df["epoch"].dropna().min()
        for i, row in enumerate(evt_df.itertuples(index=False)):
            is_root = str(row.tweet_id) == str(cid)
            if is_root:
                root_found = True

            if row.epoch is None or pd.isna(row.epoch) or pd.isna(start_epoch):
                seconds_since_start = i * 60
                event_epoch = i
            else:
                seconds_since_start = int((float(row.epoch) - float(start_epoch)) * 1000)
                if seconds_since_start < 0:
                    seconds_since_start = i * 60
                event_epoch = float(row.epoch)

            records.append(
                {
                    "tweet_id": row.tweet_id,
                    "user_id": row.user_id if row.user_id else "unknown",
                    "timestamp": row.date if row.date else f"real_tweet_{i}",
                    "epoch": event_epoch,
                    "seconds_since_start": seconds_since_start,
                    "is_root": is_root,
                    "text": row.text,
                }
            )
        if records and not root_found:
            records[0]["is_root"] = True
        normalized[cid] = records

    return normalized


def collect_responder_reply_counts(
    data_files: list[Path],
    train_conversation_ids: set[str],
    target_user_id: str,
) -> Counter[str]:
    """Count reply events per responder in training conversations."""
    wanted = [
        "id",
        "id_str",
        "conversationId",
        "conversationIdStr",
        "inReplyToTweetId",
        "in_reply_to_status_id_str",
        "user",
    ]
    responder_counts: Counter[str] = Counter()

    for path in data_files:
        usecols = read_usecols(path, wanted)
        if "user" not in usecols:
            continue

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=25000):
            conv_id = pick_column(
                chunk, ["conversationIdStr", "conversationId"]
            ).map(norm_id)
            in_reply = pick_column(
                chunk, ["in_reply_to_status_id_str", "inReplyToTweetId"]
            ).map(norm_id)
            tweet_id = pick_column(chunk, ["id_str", "id"]).map(norm_id)
            user_blob = pick_column(chunk, ["user"]).astype(str)
            user_id = user_blob.str.extract(USER_ID_RE, expand=False)

            in_train = conv_id.isin(train_conversation_ids)
            if not in_train.any():
                continue

            # Non-root rows in training conversations are reply events.
            no_parent = in_reply.isna()
            same_conv = tweet_id == conv_id
            is_root = no_parent | same_conv
            is_reply_event = in_train & (~is_root) & user_id.notna()
            if not is_reply_event.any():
                continue

            sub_users = user_id[is_reply_event]
            for uid in sub_users:
                if uid == target_user_id:
                    continue
                responder_counts[uid] += 1

    return responder_counts


def load_and_aggregate_agent_pool(processed_files: list[Path]) -> pd.DataFrame:
    """Load processed agent chunks and aggregate repeated user IDs."""
    rows: list[pd.DataFrame] = []
    wanted = [
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
    ]

    for path in processed_files:
        usecols = read_usecols(path, wanted)
        if "user_id" not in usecols:
            continue
        df = pd.read_csv(path, usecols=usecols)
        rows.append(df)

    if not rows:
        raise ValueError("No processed agent files found with expected schema.")

    pool = pd.concat(rows, ignore_index=True)
    pool["user_id"] = pd.to_numeric(pool["user_id"], errors="coerce")
    pool = pool[pool["user_id"].notna()].copy()
    pool["user_id"] = pool["user_id"].astype(np.int64)

    pool["tweet_count"] = pd.to_numeric(pool["tweet_count"], errors="coerce").fillna(1)
    pool["view_count"] = pd.to_numeric(pool["view_count"], errors="coerce").fillna(0)
    pool["reply_count"] = pd.to_numeric(pool["reply_count"], errors="coerce").fillna(0)

    numeric_cols = [
        "political_score",
        "emotion_score",
        "sentiment_score",
        "hate_score",
        "offensive_score",
    ]
    for col in numeric_cols:
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0.0)

    weight = pool["tweet_count"].clip(lower=1)
    weight_sum = weight.groupby(pool["user_id"]).sum()

    agg = pd.DataFrame(index=weight_sum.index)
    agg["tweet_count"] = pool.groupby("user_id")["tweet_count"].sum()
    agg["view_count"] = pool.groupby("user_id")["view_count"].sum()
    agg["reply_count"] = pool.groupby("user_id")["reply_count"].sum()

    for col in numeric_cols:
        weighted = (pool[col] * weight).groupby(pool["user_id"]).sum()
        agg[col] = (weighted / weight_sum).fillna(0.0)

    agg["political_label"] = pool.groupby("user_id")["political_label"].agg(stable_mode)
    agg["emotion_label"] = pool.groupby("user_id")["emotion_label"].agg(stable_mode)
    agg["sentiment_label"] = pool.groupby("user_id")["sentiment_label"].agg(stable_mode)

    agg = agg.reset_index().rename(columns={"index": "user_id"})
    return agg


def sample_agents_from_responders(
    responder_profiles: pd.DataFrame,
    responder_counts: Counter[str],
    global_pool: pd.DataFrame,
    n_agents: int,
    seed: int,
) -> pd.DataFrame:
    """Sample synthetic agent roster from responder personalities."""
    if len(responder_profiles) == 0:
        raise ValueError("Responder profile set is empty.")

    responder_profiles = responder_profiles.copy()
    responder_profiles["reply_events"] = responder_profiles["user_id"].astype(str).map(
        lambda x: responder_counts.get(x, 0)
    )
    responder_profiles = responder_profiles[responder_profiles["reply_events"] > 0].copy()
    if len(responder_profiles) == 0:
        raise ValueError("No responder profiles have reply-event weights.")

    probs = responder_profiles["reply_events"].astype(float).to_numpy()
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    replace = len(responder_profiles) < n_agents
    chosen_idx = rng.choice(
        responder_profiles.index.to_numpy(),
        size=n_agents,
        replace=replace,
        p=probs,
    )
    sampled = responder_profiles.loc[chosen_idx].copy().reset_index(drop=True)

    max_uid = int(pd.to_numeric(global_pool["user_id"], errors="coerce").max()) + 1
    sampled["source_user_id"] = sampled["user_id"]
    sampled["user_id"] = np.arange(max_uid, max_uid + len(sampled), dtype=np.int64)
    sampled = sampled[AGENT_KEEP_COLS + ["source_user_id", "reply_events"]].copy()
    return sampled


def build_run_directories(
    out_dir: Path,
    holdout_roots: pd.DataFrame,
    conversation_events: dict[str, list[dict[str, Any]]],
    sampled_agents: pd.DataFrame,
    target_user_id: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    max_rounds: int,
) -> list[dict[str, Any]]:
    """Create run folders for each holdout root, with 0-shot configs."""
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_entries: list[dict[str, Any]] = []
    for i, row in enumerate(holdout_roots.itertuples(index=False), start=1):
        tweet_id = str(row.tweet_id)
        conv_id = str(row.conversation_id)
        run_name = f"holdout_{i:02d}_{tweet_id}"
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        agents_path = run_dir / "agents_for_thread.csv"
        sampled_agents[AGENT_KEEP_COLS].to_csv(agents_path, index=False)

        root_text = str(row.text)
        root_event = {
            "tweet_id": tweet_id,
            "user_id": target_user_id,
            "timestamp": "target_root_only",
            "epoch": 0,
            "seconds_since_start": 0,
            "is_root": True,
            "text": root_text,
        }

        sim_metadata = {
            "root_tweet": {
                "id": tweet_id,
                "conversation_id": conv_id,
                "user_id": target_user_id,
                "text": root_text,
                "expected_replies": int(row.reply_count),
            },
            "temporal_events": [root_event],
            "source": "forward_holdout_zero_shot",
        }
        sim_metadata_path = run_dir / "thread_metadata.json"
        with open(sim_metadata_path, "w") as f:
            json.dump(sim_metadata, f, indent=2)

        real_events = conversation_events.get(conv_id, [])
        real_metadata = {
            "root_tweet": {
                "id": tweet_id,
                "conversation_id": conv_id,
                "user_id": target_user_id,
                "text": root_text,
                "expected_replies": max(len(real_events) - 1, 0),
            },
            "temporal_events": real_events,
            "source": "real_holdout_from_dataset",
        }
        real_metadata_path = run_dir / "real_thread_metadata.json"
        with open(real_metadata_path, "w") as f:
            json.dump(real_metadata, f, indent=2)

        config = {
            "target_tweet_id": tweet_id,
            "paths": {
                "thread_metadata": str(sim_metadata_path.resolve()),
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
                "model": llm_model,
                "api_key_env": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "simulation": {
                "max_rounds": max_rounds,
                "thread_num": i,
                "use_few_shot": False,
            },
        }
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        run_entries.append(
            {
                "run_name": run_name,
                "tweet_id": tweet_id,
                "conversation_id": conv_id,
                "run_dir": str(run_dir.resolve()),
                "config_path": str(config_path.resolve()),
                "real_metadata_path": str(real_metadata_path.resolve()),
                "sim_metadata_path": str(sim_metadata_path.resolve()),
                "actual_root_reply_count": int(row.reply_count),
            }
        )

    with open(out_dir / "runs_manifest.json", "w") as f:
        json.dump({"runs": run_entries}, f, indent=2)
    return run_entries


def run_personality_phase(args: argparse.Namespace) -> None:
    data_files = sorted((PROJECT_ROOT / "data").glob(args.data_glob))
    if not data_files:
        raise FileNotFoundError(f"No data files found for glob: {args.data_glob}")

    processed_files = sorted((PROJECT_ROOT / "processed_agents").glob(args.processed_glob))
    if not processed_files:
        raise FileNotFoundError(
            f"No processed-agent files found for glob: {args.processed_glob}"
        )

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_uid = str(args.target_user_id)
    roots, username = find_target_root_tweets(data_files, target_uid)
    if len(roots) < (args.train_roots + args.holdout_roots):
        raise ValueError(
            f"Need at least {args.train_roots + args.holdout_roots} roots, "
            f"found {len(roots)}"
        )

    train = roots.head(args.train_roots).copy()
    holdout = roots.tail(args.holdout_roots).copy().reset_index(drop=True)

    train_conv_ids = set(train["conversation_id"].dropna().astype(str))
    holdout_conv_ids = set(holdout["conversation_id"].dropna().astype(str))
    convo_events = collect_conversation_events(data_files, holdout_conv_ids | train_conv_ids)

    responder_counts = collect_responder_reply_counts(
        data_files=data_files,
        train_conversation_ids=train_conv_ids,
        target_user_id=target_uid,
    )

    pool = load_and_aggregate_agent_pool(processed_files)
    responder_ids = {int(uid) for uid in responder_counts.keys()}
    responder_profiles = pool[pool["user_id"].isin(responder_ids)].copy()
    if len(responder_profiles) == 0:
        raise ValueError("No responder profiles could be matched in processed agent pool.")

    sampled_agents = sample_agents_from_responders(
        responder_profiles=responder_profiles,
        responder_counts=responder_counts,
        global_pool=pool,
        n_agents=args.target_agents,
        seed=args.seed,
    )

    train.to_csv(out_dir / "training_roots.csv", index=False)
    holdout.to_csv(out_dir / "holdout_roots.csv", index=False)
    responder_profiles.to_csv(out_dir / "responder_profiles.csv", index=False)
    sampled_agents.to_csv(out_dir / "sampled_agents.csv", index=False)

    run_entries = build_run_directories(
        out_dir=out_dir,
        holdout_roots=holdout,
        conversation_events=convo_events,
        sampled_agents=sampled_agents,
        target_user_id=target_uid,
        llm_model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
    )

    manifest = {
        "target_user_id": target_uid,
        "target_username": username,
        "total_root_tweets_found": int(len(roots)),
        "train_roots": int(len(train)),
        "holdout_roots": int(len(holdout)),
        "train_reply_count_mean": float(train["reply_count"].mean()),
        "train_reply_count_median": float(train["reply_count"].median()),
        "responder_unique_users": int(len(responder_counts)),
        "responder_reply_events": int(sum(responder_counts.values())),
        "matched_responder_profiles": int(len(responder_profiles)),
        "sampled_agents": int(len(sampled_agents)),
        "model": args.model,
        "use_few_shot": False,
        "runs_prepared": len(run_entries),
        "output_dir": str(out_dir.resolve()),
    }

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("PERSONALITY PHASE COMPLETE")
    print("=" * 80)
    print(f"Target account: {username} ({target_uid})")
    print(f"Root tweets found: {len(roots)}")
    print(f"Training roots: {len(train)} | Holdout roots: {len(holdout)}")
    print(f"Unique responders in training roots: {len(responder_counts)}")
    print(f"Responder profiles matched: {len(responder_profiles)}")
    print(f"Sampled agents: {len(sampled_agents)}")
    print(f"Prepared holdout runs: {len(run_entries)}")
    print(f"Output: {out_dir}")
    print("=" * 80)


def run_simulation_phase(args: argparse.Namespace) -> None:
    from sim.thread_simulation import ThreadModel

    out_dir = PROJECT_ROOT / args.output_dir
    manifest_path = out_dir / "runs_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing runs manifest: {manifest_path}. Run personalities phase first."
        )

    with open(manifest_path) as f:
        runs = json.load(f)["runs"]

    print("=" * 80)
    print("SIMULATION PHASE START")
    print("=" * 80)
    print(f"Runs: {len(runs)}")

    summary: list[dict[str, Any]] = []
    for idx, run in enumerate(runs, start=1):
        run_name = run["run_name"]
        run_dir = Path(run["run_dir"])
        config_path = Path(run["config_path"])
        sim_output = run_dir / "simulation_output"
        sim_output.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(runs)}] {run_name}")
        model = ThreadModel(config_path=str(config_path))
        model.run(max_rounds=args.max_rounds)
        model.export_results(output_dir=str(sim_output))

        summary.append(
            {
                "run_name": run_name,
                "tweet_id": run["tweet_id"],
                "actual_root_reply_count": run["actual_root_reply_count"],
                "sim_total_posts": len(model.thread_history),
                "sim_replies": len(model.thread_history) - 1,
                "sim_max_depth": max(p["depth"] for p in model.thread_history),
            }
        )

    summary_path = out_dir / "simulation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"runs": summary}, f, indent=2)

    print("\n" + "=" * 80)
    print("SIMULATION PHASE COMPLETE")
    print("=" * 80)
    print(f"Saved: {summary_path}")


def run_analysis_phase(args: argparse.Namespace) -> None:
    from scripts.validate_thread_simulation import ThreadValidator

    out_dir = PROJECT_ROOT / args.output_dir
    manifest_path = out_dir / "runs_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing runs manifest: {manifest_path}. Run personalities phase first."
        )

    with open(manifest_path) as f:
        runs = json.load(f)["runs"]

    validator = ThreadValidator(device=args.device)

    label_val = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    records: list[dict[str, Any]] = []

    for idx, run in enumerate(runs, start=1):
        run_name = run["run_name"]
        run_dir = Path(run["run_dir"])
        real_path = Path(run["real_metadata_path"])
        sim_path = run_dir / "simulation_output" / "simulated_thread_metadata.json"
        if not sim_path.exists():
            print(f"[{idx}/{len(runs)}] {run_name}: skipped (missing simulation output)")
            continue

        print(f"[{idx}/{len(runs)}] {run_name}: analyzing")
        real_tweets = validator.load_thread(str(real_path))
        sim_tweets = validator.load_thread(str(sim_path))
        real_cls = validator.classify_sentiment(real_tweets)
        sim_cls = validator.classify_sentiment(sim_tweets)

        real_dist = validator.calculate_sentiment_distribution(real_cls)
        sim_dist = validator.calculate_sentiment_distribution(sim_cls)
        sent_jsd = validator.calculate_jsd(real_dist, sim_dist)

        real_mean = float(np.mean([label_val[t["sentiment_label"]] for t in real_cls]))
        sim_mean = float(np.mean([label_val[t["sentiment_label"]] for t in sim_cls]))
        residual = sim_mean - real_mean

        records.append(
            {
                "run_name": run_name,
                "tweet_id": run["tweet_id"],
                "actual_root_reply_count": run["actual_root_reply_count"],
                "sim_replies": len(sim_tweets) - 1,
                "reply_count_error": (len(sim_tweets) - 1) - run["actual_root_reply_count"],
                "sentiment_jsd": sent_jsd,
                "sentiment_mean_real": real_mean,
                "sentiment_mean_sim": sim_mean,
                "sentiment_residual": residual,
                "real_negative_pct": real_dist.get("Negative", 0.0),
                "sim_negative_pct": sim_dist.get("Negative", 0.0),
            }
        )

    if not records:
        raise ValueError("No analysis records produced.")

    df = pd.DataFrame(records)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(analysis_dir / "holdout_results.csv", index=False)

    summary = {
        "n_runs": int(len(df)),
        "mean_sentiment_jsd": float(df["sentiment_jsd"].mean()),
        "std_sentiment_jsd": float(df["sentiment_jsd"].std(ddof=0)),
        "mean_sentiment_residual": float(df["sentiment_residual"].mean()),
        "std_sentiment_residual": float(df["sentiment_residual"].std(ddof=0)),
        "mean_reply_count_error": float(df["reply_count_error"].mean()),
        "std_reply_count_error": float(df["reply_count_error"].std(ddof=0)),
    }
    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("ANALYSIS PHASE COMPLETE")
    print("=" * 80)
    print(f"Runs analyzed: {len(df)}")
    print(f"Mean sentiment JSD: {summary['mean_sentiment_jsd']:.4f}")
    print(f"Mean sentiment residual: {summary['mean_sentiment_residual']:.4f}")
    print(f"Mean reply count error: {summary['mean_reply_count_error']:.2f}")
    print(f"Saved: {analysis_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward prediction pipeline for one account."
    )
    parser.add_argument(
        "phase",
        choices=["personalities", "simulate", "analyze"],
        help="Pipeline phase to run.",
    )
    parser.add_argument("--target-user-id", type=str, default="537709549")
    parser.add_argument("--train-roots", type=int, default=35)
    parser.add_argument("--holdout-roots", type=int, default=5)
    parser.add_argument("--target-agents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="dolphin-llama3:8b")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-glob", type=str, default="may_july_chunk_*.csv")
    parser.add_argument(
        "--processed-glob", type=str, default="processed_agents_chunk_*.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/future_prediction/laura_forward",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "personalities":
        run_personality_phase(args)
    elif args.phase == "simulate":
        run_simulation_phase(args)
    elif args.phase == "analyze":
        run_analysis_phase(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()

