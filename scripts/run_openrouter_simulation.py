from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openrouter_runtime.client import OpenRouterClient
from openrouter_runtime.simulation import (
    OpenRouterThreadSimulation,
    load_simulation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a rate-aware OpenRouter simulation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/openrouter_smoke_config.yaml"),
        help="YAML config describing the root post and agent personas.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the configured number of rounds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openrouter_output"),
        help="Output directory for the run artifacts.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=20,
        help="Global request budget for the OpenRouter client.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max in-flight OpenRouter requests within a round.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional OpenRouter model override.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Retry budget for upstream provider throttling.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    output_root = PROJECT_ROOT / args.output_dir

    config = load_simulation_config(config_path)
    if args.model:
        config.setdefault("llm", {})
        config["llm"]["model"] = args.model
    configured_model = config.get("llm", {}).get("model")
    rounds = args.rounds or config.get("simulation", {}).get("max_rounds", 1)
    client = OpenRouterClient.from_env(
        PROJECT_ROOT,
        requests_per_minute=args.rpm,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
    )
    if configured_model:
        client.model = configured_model
    key_status = client.key_status()
    model_info = client.ensure_model_available()

    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "key_status.json").write_text(
        json.dumps(key_status, indent=2),
        encoding="utf-8",
    )
    (run_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2),
        encoding="utf-8",
    )

    simulation = OpenRouterThreadSimulation(config=config, client=client)
    try:
        posts = simulation.run(rounds=rounds)
        output_path = simulation.export_results(run_dir)
        summary = {
            "status": "success",
            "rounds": rounds,
            "total_posts": len(posts),
            "output_path": str(output_path),
            "model": client.model,
            "parallel_workers": simulation.parallel_workers,
            "rpm": args.rpm,
            "max_concurrency": args.concurrency,
        }
    except Exception as exc:
        summary = {
            "status": "error",
            "rounds": rounds,
            "model": client.model,
            "rpm": args.rpm,
            "max_concurrency": args.concurrency,
            "error": str(exc),
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
