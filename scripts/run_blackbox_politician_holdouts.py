#!/usr/bin/env python3
"""
Strict forward-facing holdout runner for politician accounts.

For each target account:
1. Find all root tweets in chronological order.
2. Use every root except the latest as training history.
3. Build a responder personality pool from those prior conversations only.
4. Treat the latest root tweet as a black-box holdout.
5. Run one zero-shot simulation using only the holdout root text.
6. Compare simulated vs real holdout thread using sentiment validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.laura_forward_pipeline import (  # noqa: E402
    AGENT_KEEP_COLS,
    collect_conversation_events,
    collect_responder_reply_counts,
    find_target_root_tweets,
    load_and_aggregate_agent_pool,
    sample_agents_from_responders,
)
from scripts.validate_thread_simulation import ThreadValidator  # noqa: E402
from sim.thread_simulation import ThreadModel  # noqa: E402


DEFAULT_TARGETS: list[dict[str, str]] = [
    {"name": "Joe Biden", "user_id": "939091"},
    {"name": "Ted Cruz", "user_id": "23022687"},
    {"name": "Sen. Marsha Blackburn", "user_id": "278145569"},
    {"name": "Senator Marco Rubio", "user_id": "229966028"},
    {"name": "Rep. Brian Mast", "user_id": "814103950404239360"},
]


def build_run_files(
    run_dir: Path,
    target_name: str,
    target_user_id: str,
    holdout_row: pd.Series,
    conversation_events: list[dict[str, Any]],
    sampled_agents: pd.DataFrame,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_rounds: int,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)

    agents_path = run_dir / "agents_for_thread.csv"
    sampled_agents[AGENT_KEEP_COLS].to_csv(agents_path, index=False)

    tweet_id = str(holdout_row["tweet_id"])
    conv_id = str(holdout_row["conversation_id"])
    root_text = str(holdout_row["text"])

    sim_metadata = {
        "root_tweet": {
            "id": tweet_id,
            "conversation_id": conv_id,
            "user_id": target_user_id,
            "text": root_text,
            "expected_replies": int(holdout_row["reply_count"]),
            "target_name": target_name,
        },
        "temporal_events": [
            {
                "tweet_id": tweet_id,
                "user_id": target_user_id,
                "timestamp": "target_root_only",
                "epoch": 0,
                "seconds_since_start": 0,
                "is_root": True,
                "text": root_text,
            }
        ],
        "source": "black_box_latest_root_holdout",
    }
    sim_metadata_path = run_dir / "thread_metadata.json"
    sim_metadata_path.write_text(json.dumps(sim_metadata, indent=2))

    real_metadata = {
        "root_tweet": {
            "id": tweet_id,
            "conversation_id": conv_id,
            "user_id": target_user_id,
            "text": root_text,
            "expected_replies": max(len(conversation_events) - 1, 0),
            "target_name": target_name,
        },
        "temporal_events": conversation_events,
        "source": "real_latest_root_holdout",
    }
    real_metadata_path = run_dir / "real_thread_metadata.json"
    real_metadata_path.write_text(json.dumps(real_metadata, indent=2))

    llm_cfg: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "openrouter":
        llm_cfg["api_key_env"] = "OPENROUTER_API_KEY"
        llm_cfg["base_url"] = "https://openrouter.ai/api/v1"
        llm_cfg["site_url"] = "https://local.simulation.run"
        llm_cfg["app_name"] = "Dissertation_Final"

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
        "llm": llm_cfg,
        "simulation": {
            "max_rounds": max_rounds,
            "thread_num": 1,
            "use_few_shot": False,
        },
    }
    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    return {
        "agents_path": agents_path,
        "sim_metadata_path": sim_metadata_path,
        "real_metadata_path": real_metadata_path,
        "config_path": config_path,
    }


def summarize_sentiment(
    validator: ThreadValidator,
    real_path: Path,
    sim_path: Path,
) -> dict[str, float]:
    real_cls = validator.classify_sentiment(validator.load_thread(real_path))
    sim_cls = validator.classify_sentiment(validator.load_thread(sim_path))

    real_dist = validator.calculate_sentiment_distribution(real_cls)
    sim_dist = validator.calculate_sentiment_distribution(sim_cls)
    jsd = validator.calculate_jsd(real_dist, sim_dist)

    label_val = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    real_mean = float(sum(label_val[t["sentiment_label"]] for t in real_cls) / len(real_cls))
    sim_mean = float(sum(label_val[t["sentiment_label"]] for t in sim_cls) / len(sim_cls))

    return {
        "sentiment_jsd": float(jsd),
        "sentiment_mean_real": real_mean,
        "sentiment_mean_sim": sim_mean,
        "sentiment_residual": sim_mean - real_mean,
        "real_negative_pct": float(real_dist.get("Negative", 0.0)),
        "real_neutral_pct": float(real_dist.get("Neutral", 0.0)),
        "real_positive_pct": float(real_dist.get("Positive", 0.0)),
        "sim_negative_pct": float(sim_dist.get("Negative", 0.0)),
        "sim_neutral_pct": float(sim_dist.get("Neutral", 0.0)),
        "sim_positive_pct": float(sim_dist.get("Positive", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict latest-root black-box holdouts for politician accounts."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/future_prediction/politician_blackbox_openrouter",
    )
    parser.add_argument("--provider", type=str, default="openrouter")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.3-70b-instruct",
    )
    parser.add_argument("--target-agents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-glob", type=str, default="may_july_chunk_*.csv")
    parser.add_argument(
        "--processed-glob",
        type=str,
        default="processed_agents_chunk_*.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    print("=" * 80)
    print("LOADING GLOBAL AGENT POOL")
    print("=" * 80)
    pool = load_and_aggregate_agent_pool(processed_files)
    print(f"Loaded agent pool: {len(pool)} unique users")

    validator = ThreadValidator(device=args.device)
    records: list[dict[str, Any]] = []

    for idx, target in enumerate(DEFAULT_TARGETS, start=1):
        target_name = target["name"]
        target_uid = target["user_id"]
        account_dir = out_dir / f"{idx:02d}_{target_name.lower().replace(' ', '_').replace('.', '').replace('-', '_')}"

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(DEFAULT_TARGETS)}] {target_name} ({target_uid})")
        print("=" * 80)

        roots, username = find_target_root_tweets(data_files, target_uid)
        if len(roots) < 2:
            raise ValueError(f"{target_name}: need at least 2 root tweets, found {len(roots)}")

        train = roots.head(len(roots) - 1).copy()
        holdout = roots.tail(1).copy().reset_index(drop=True)
        holdout_row = holdout.iloc[0]

        train_conv_ids = set(train["conversation_id"].dropna().astype(str))
        holdout_conv_id = str(holdout_row["conversation_id"])
        convo_events = collect_conversation_events(
            data_files,
            train_conv_ids | {holdout_conv_id},
        )
        holdout_events = convo_events.get(holdout_conv_id, [])

        responder_counts = collect_responder_reply_counts(
            data_files=data_files,
            train_conversation_ids=train_conv_ids,
            target_user_id=target_uid,
        )
        responder_ids = {int(uid) for uid in responder_counts.keys()}
        responder_profiles = pool[pool["user_id"].isin(responder_ids)].copy()
        if len(responder_profiles) == 0:
            raise ValueError(f"{target_name}: no responder profiles matched in processed pool")

        sampled_agents = sample_agents_from_responders(
            responder_profiles=responder_profiles,
            responder_counts=responder_counts,
            global_pool=pool,
            n_agents=args.target_agents,
            seed=args.seed + idx,
        )

        files = build_run_files(
            run_dir=account_dir,
            target_name=target_name,
            target_user_id=target_uid,
            holdout_row=holdout_row,
            conversation_events=holdout_events,
            sampled_agents=sampled_agents,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_rounds=args.max_rounds,
        )

        model = ThreadModel(config_path=str(files["config_path"]))
        model.run(max_rounds=args.max_rounds)
        sim_output = account_dir / "simulation_output"
        sim_output.mkdir(parents=True, exist_ok=True)
        model.export_results(output_dir=str(sim_output))

        sim_metadata_path = sim_output / "simulated_thread_metadata.json"
        sentiment = summarize_sentiment(
            validator=validator,
            real_path=files["real_metadata_path"],
            sim_path=sim_metadata_path,
        )

        record = {
            "target_name": target_name,
            "target_user_id": target_uid,
            "target_username": username,
            "total_roots_found": int(len(roots)),
            "training_roots": int(len(train)),
            "holdout_tweet_id": str(holdout_row["tweet_id"]),
            "holdout_date": (
                holdout_row["date"].isoformat()
                if pd.notna(holdout_row["date"])
                else None
            ),
            "holdout_reply_count_real": int(max(len(holdout_events) - 1, 0)),
            "holdout_reply_count_reported": int(holdout_row["reply_count"]),
            "sim_replies": int(len(model.thread_history) - 1),
            "sim_total_posts": int(len(model.thread_history)),
            "sim_max_depth": int(max(p["depth"] for p in model.thread_history)),
            "root_text": str(holdout_row["text"]),
            "run_dir": str(account_dir.resolve()),
            **sentiment,
        }
        records.append(record)

        summary_path = account_dir / "result_summary.json"
        summary_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_dir / "all_results.csv", index=False)

    aggregate = {
        "n_accounts": int(len(records)),
        "provider": args.provider,
        "model": args.model,
        "mean_sentiment_jsd": float(results_df["sentiment_jsd"].mean()),
        "mean_sentiment_residual": float(results_df["sentiment_residual"].mean()),
        "mean_reply_count_error_vs_real": float(
            (results_df["sim_replies"] - results_df["holdout_reply_count_real"]).mean()
        ),
        "mean_real_negative_pct": float(results_df["real_negative_pct"].mean()),
        "mean_sim_negative_pct": float(results_df["sim_negative_pct"].mean()),
    }
    (out_dir / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2))

    print("\n" + "=" * 80)
    print("BLACK-BOX HOLDOUT RUN COMPLETE")
    print("=" * 80)
    print(json.dumps(aggregate, indent=2))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
