from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an OpenRouter simulation config from a reconstructed thread."
    )
    parser.add_argument(
        "--thread-dir",
        type=Path,
        required=True,
        help="Path to batch_simulations_reconstructed/thread_NNN",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output YAML path.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=200,
        help="Target number of agent personas.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Max rounds for the simulation.",
    )
    parser.add_argument(
        "--max-posts-per-round",
        type=int,
        default=10,
        help="Cap generated posts per round.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Within-round OpenRouter workers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.3-70b-instruct",
        help="OpenRouter model to embed in the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Sampling seed for deterministic persona upsampling.",
    )
    return parser.parse_args()


def map_aggression(row: dict[str, str]) -> float:
    offensive = float(row.get("offensive_score", 0.2) or 0.2)
    hate = float(row.get("hate_score", 0.0) or 0.0)
    sentiment = row.get("sentiment_label", "neutral").lower()
    sentiment_bonus = 0.08 if sentiment == "negative" else 0.0
    return max(0.05, min(0.95, max(offensive, hate) + sentiment_bonus))


def build_agents(rows: list[dict[str, str]], target_count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for idx in range(target_count):
        row = rows[idx] if idx < len(rows) else rng.choice(rows)
        political = row.get("political_label", "Center") or "Center"
        selected.append(
            {
                "user_id": f"{row.get('user_id', 'user')}_{idx:03d}",
                "display_name": f"{political.lower()}_{idx:03d}",
                "political_label": political,
                "aggression": round(map_aggression(row), 3),
                "emotion": row.get("emotion_label", "anger") or "anger",
            }
        )
    return selected


def main() -> None:
    args = parse_args()
    thread_dir = args.thread_dir
    metadata_path = thread_dir / "thread_metadata.json"
    agents_path = thread_dir / "agents_for_thread.csv"

    metadata = json.loads(metadata_path.read_text())
    rows = list(csv.DictReader(agents_path.open()))
    agents = build_agents(rows, args.agents, args.seed)

    payload = {
        "llm": {
            "model": args.model,
            "temperature": 0.8,
            "max_tokens": 110,
        },
        "simulation": {
            "max_rounds": args.rounds,
            "max_posts_per_round": args.max_posts_per_round,
            "parallel_workers": args.parallel_workers,
        },
        "root_post": {
            "author_id": metadata["root_tweet"].get("user_id", "root_user"),
            "author_name": f"thread_{thread_dir.name}_root",
            "text": metadata["root_tweet"]["text"],
        },
        "agents": agents,
        "comparison_source": {
            "thread_dir": str(thread_dir),
            "real_thread_metadata": str(metadata_path),
            "actual_reply_count": metadata["root_tweet"].get("expected_replies"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
