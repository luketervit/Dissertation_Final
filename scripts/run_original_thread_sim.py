from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim.thread_simulation import ThreadModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original ThreadModel with a supplied config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ThreadModel(config_path=str(args.config))
    model.run(max_rounds=args.rounds)
    model.export_results(output_dir=str(args.output_dir))

    payload = {
        "status": "success",
        "rounds": args.rounds,
        "total_posts": len(model.thread_history),
        "output_dir": str(args.output_dir),
        "model": model.config["llm"]["model"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
