from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        description="Run multiple OpenRouter simulations with shared throttling."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config/openrouter_batch"),
        help="Directory containing per-run YAML configs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel simulation workers sharing one rate limiter.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=20,
        help="Shared request budget across all workers.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max in-flight OpenRouter requests across the batch.",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openrouter_batch_output"),
        help="Destination for batch outputs.",
    )
    return parser.parse_args()


def run_config(config_path: Path, run_dir: Path, client: OpenRouterClient) -> dict:
    config = load_simulation_config(config_path)
    if client.model:
        config.setdefault("llm", {})
        config["llm"]["model"] = client.model
    rounds = config.get("simulation", {}).get("max_rounds", 1)
    sim = OpenRouterThreadSimulation(config=config, client=client)
    posts = sim.run(rounds=rounds)
    output_path = sim.export_results(run_dir)
    return {
        "config": str(config_path),
        "rounds": rounds,
        "total_posts": len(posts),
        "output_path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    client = OpenRouterClient.from_env(
        PROJECT_ROOT,
        requests_per_minute=args.rpm,
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout,
    )
    if args.model:
        client.model = args.model
    client.ensure_model_available()
    batch_root = PROJECT_ROOT / args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_root.mkdir(parents=True, exist_ok=True)

    config_dir = PROJECT_ROOT / args.config_dir
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"No YAML configs found in {config_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {}
        for config_path in config_paths:
            run_dir = batch_root / config_path.stem
            future = pool.submit(run_config, config_path, run_dir, client)
            future_map[future] = config_path

        for future in as_completed(future_map):
            config_path = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "config": str(config_path),
                        "error": str(exc),
                    }
                )

    summary_path = batch_root / "batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"batch_summary": str(summary_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
