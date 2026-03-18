from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare original ThreadModel assets for an OpenRouter rerun."
    )
    parser.add_argument("--thread-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.3-70b-instruct",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_agents(rows: list[dict[str, str]], target_count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    expanded = []
    for idx in range(target_count):
        base = dict(rows[idx] if idx < len(rows) else rng.choice(rows))
        base["user_id"] = f"{base['user_id']}_{idx:03d}"
        expanded.append(base)
    return expanded


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_agents = list(csv.DictReader((args.thread_dir / "agents_for_thread.csv").open()))
    expanded_agents = build_agents(source_agents, args.agents, args.seed)

    agents_out = args.output_dir / "agents_for_thread_200.csv"
    with agents_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expanded_agents[0].keys())
        writer.writeheader()
        writer.writerows(expanded_agents)

    metadata = json.loads((args.thread_dir / "thread_metadata.json").read_text())
    metadata_out = args.output_dir / "thread_metadata.json"
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    config = {
        "target_tweet_id": metadata["root_tweet"]["id"],
        "paths": {
            "thread_metadata": str(metadata_out),
            "agents_for_thread": str(agents_out),
        },
        "abm": {
            "lurker_ratio": 0,
            "lurker_dist_strategy": "match_active_agents",
            "bounded_confidence_threshold": 0.3,
            "backfire_threshold": 0.6,
            "backfire_aggression_min": 0.7,
        },
        "llm": {
            "provider": "openrouter",
            "model": args.model,
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "site_url": "https://local.simulation.run",
            "app_name": "Dissertation_Final",
            "temperature": 0.9,
            "max_tokens": 150,
        },
        "simulation": {
            "max_rounds": 10,
            "thread_num": 1,
            "use_few_shot": True,
        },
    }
    config_out = args.output_dir / "config.yaml"
    config_out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(config_out)


if __name__ == "__main__":
    main()
