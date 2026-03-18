from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten OpenRouter simulation output into validator list format."
    )
    parser.add_argument("--input", type=Path, required=True, help="simulated_thread.json")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON list path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    posts = data.get("posts", [])
    flattened = [
        {
            "tweet_id": post.get("post_id"),
            "post_id": post.get("post_id"),
            "text": post.get("text", ""),
        }
        for post in posts
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(flattened, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
