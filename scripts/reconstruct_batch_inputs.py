"""
Reconstruct batch simulation input directories from previous run artifacts.

Reads:
  - batch_analysis_100/all_tweets_classified.csv (real tweet texts + thread_id)
  - batch_output_100/thread_NNN/agents_for_thread.csv (agent DNA profiles)

Creates for each thread:
  - thread_metadata.json (root tweet + real tweet texts for few-shot)
  - config.yaml (simulation configuration)
  - agents_for_thread.csv (copied from batch_output_100)

Usage:
    python scripts/reconstruct_batch_inputs.py \
        --output batch_simulations_reconstructed
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct batch simulation inputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_simulations_reconstructed",
        help="Output directory for reconstructed inputs",
    )
    parser.add_argument(
        "--classified-csv",
        type=str,
        default="batch_analysis_100/all_tweets_classified.csv",
        help="Path to classified tweets CSV",
    )
    parser.add_argument(
        "--agents-dir",
        type=str,
        default="batch_output_100",
        help="Directory containing thread_NNN/agents_for_thread.csv",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output
    classified_path = project_root / args.classified_csv
    agents_dir = project_root / args.agents_dir

    print("=" * 80)
    print("RECONSTRUCT BATCH SIMULATION INPUTS")
    print("=" * 80)
    print(f"  Classified CSV: {classified_path}")
    print(f"  Agents dir:     {agents_dir}")
    print(f"  Output dir:     {output_dir}")

    # Load classified tweets
    print("\nLoading classified tweets...")
    df = pd.read_csv(classified_path)
    real_df = df[df["source"] == "real"].copy()
    thread_ids = sorted(real_df["thread_id"].unique())
    print(f"  Found {len(thread_ids)} threads, {len(real_df)} real tweets")

    output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    for thread_id in thread_ids:
        thread_dir = output_dir / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)

        # Get real tweets for this thread
        thread_tweets = real_df[real_df["thread_id"] == thread_id].copy()
        thread_tweets = thread_tweets.sort_values("tweet_id")

        # Use first tweet as root (best approximation)
        root_row = thread_tweets.iloc[0]
        root_text = str(root_row["text"])

        # Copy agents_for_thread.csv from old output
        src_agents = agents_dir / thread_id / "agents_for_thread.csv"
        dst_agents = thread_dir / "agents_for_thread.csv"
        if src_agents.exists():
            shutil.copy2(src_agents, dst_agents)
        else:
            print(f"  WARNING: {src_agents} not found, skipping {thread_id}")
            skipped += 1
            continue

        # Build thread_metadata.json
        # Include all real tweet texts as temporal_events for few-shot
        temporal_events = []
        for idx, (_, row) in enumerate(thread_tweets.iterrows()):
            temporal_events.append(
                {
                    "tweet_id": int(row["tweet_id"])
                    if pd.notna(row["tweet_id"])
                    else idx,
                    "user_id": "unknown",
                    "timestamp": f"real_tweet_{idx}",
                    "epoch": idx,
                    "seconds_since_start": idx * 60,
                    "is_root": idx == 0,
                    "text": str(row["text"]),
                }
            )

        metadata = {
            "root_tweet": {
                "id": str(root_row["tweet_id"]),
                "conversation_id": str(root_row["tweet_id"]),
                "user_id": "unknown",
                "text": root_text,
                "expected_replies": len(thread_tweets) - 1,
            },
            "replies": {
                "actual_count": len(thread_tweets) - 1,
                "tweet_ids": [
                    int(r["tweet_id"])
                    for _, r in thread_tweets.iloc[1:].iterrows()
                    if pd.notna(r["tweet_id"])
                ],
            },
            "temporal_events": temporal_events,
            "source": "reconstructed_from_all_tweets_classified",
        }

        metadata_path = thread_dir / "thread_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Build config.yaml
        config = {
            "target_tweet_id": str(root_row["tweet_id"]),
            "paths": {
                "thread_metadata": str(metadata_path),
                "agents_for_thread": str(dst_agents),
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
                "model": "dolphin-llama3:8b",
                "api_key_env": None,
                "temperature": 0.9,
                "max_tokens": 150,
            },
            "simulation": {
                "max_rounds": 10,
                "thread_num": int(thread_id.split("_")[1]),
            },
        }

        config_path = thread_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        created += 1

    # Save manifest
    manifest = {
        "total_threads": created,
        "skipped": skipped,
        "source": "reconstructed from batch_analysis_100",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"RECONSTRUCTION COMPLETE: {created} threads, {skipped} skipped")
    print(f"Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
