"""
Test 3 threads with 2 models (dolphin-llama3:8b vs llama3.1:8b).

Runs each thread with each model (5 rounds, 30 agents), saves output
to test_3threads/results_<model>/, and prints a comparison summary.

Usage:
    python scripts/test_3thread_comparison.py
"""
import json
import os
import sys
import time
import shutil
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = ["dolphin-llama3:8b", "llama3.1:8b"]
THREAD_DIR = Path("test_3threads")
THREADS = sorted(THREAD_DIR.glob("thread_*"))
MAX_ROUNDS = 5


def run_simulation(thread_dir: Path, model: str, output_dir: Path) -> dict:
    """Run one simulation and return basic metrics."""
    from sim.thread_simulation import ThreadModel

    # Patch config with the target model
    config_path = thread_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["llm"]["model"] = model

    # Write patched config to temp location
    patched_config = output_dir / "config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(patched_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    t0 = time.time()
    model_instance = ThreadModel(config_path=str(patched_config))
    model_instance.run(max_rounds=MAX_ROUNDS)
    model_instance.export_results(output_dir=str(output_dir))
    elapsed = time.time() - t0

    # Collect basic stats from generated tweets
    posts = model_instance.thread_history[1:]  # exclude root
    texts = [p["text"] for p in posts]

    return {
        "model": model,
        "thread": thread_dir.name,
        "total_posts": len(posts),
        "max_depth": max(p["depth"] for p in model_instance.thread_history),
        "elapsed_s": round(elapsed, 1),
        "sample_tweets": texts[:5],
        "mean_aggression": (
            sum(p["aggression"] for p in posts) / len(posts) if posts else 0
        ),
        "left_pct": (
            sum(1 for p in posts if p.get("political_label") == "Left")
            / len(posts) * 100
            if posts
            else 0
        ),
        "right_pct": (
            sum(1 for p in posts if p.get("political_label") == "Right")
            / len(posts) * 100
            if posts
            else 0
        ),
    }


def main():
    print("=" * 80)
    print("3-THREAD MODEL COMPARISON TEST")
    print("=" * 80)
    print(f"Models: {MODELS}")
    print(f"Threads: {[t.name for t in THREADS]}")
    print(f"Rounds: {MAX_ROUNDS}")
    print()

    all_results = []

    for model in MODELS:
        print(f"\n{'#' * 80}")
        print(f"MODEL: {model}")
        print(f"{'#' * 80}")

        for thread_dir in THREADS:
            output_dir = THREAD_DIR / f"results_{model.replace(':', '_')}" / thread_dir.name
            print(f"\n--- {thread_dir.name} with {model} ---")

            try:
                result = run_simulation(thread_dir, model, output_dir)
                all_results.append(result)
                print(f"  Posts: {result['total_posts']}, "
                      f"Depth: {result['max_depth']}, "
                      f"Time: {result['elapsed_s']}s")
                print(f"  Left: {result['left_pct']:.0f}%, "
                      f"Right: {result['right_pct']:.0f}%, "
                      f"Mean agg: {result['mean_aggression']:.3f}")
                print(f"  Sample tweets:")
                for t in result["sample_tweets"][:3]:
                    print(f"    \"{t[:100]}\"")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model,
                    "thread": thread_dir.name,
                    "error": str(e),
                })

    # Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for thread_dir in THREADS:
        print(f"\n{thread_dir.name}:")
        for model in MODELS:
            r = next(
                (x for x in all_results
                 if x.get("thread") == thread_dir.name and x.get("model") == model),
                None,
            )
            if r and "error" not in r:
                print(
                    f"  {model:25s} | Posts: {r['total_posts']:3d} | "
                    f"L/R: {r['left_pct']:4.0f}/{r['right_pct']:4.0f}% | "
                    f"Agg: {r['mean_aggression']:.3f} | "
                    f"Time: {r['elapsed_s']}s"
                )
            elif r:
                print(f"  {model:25s} | ERROR: {r['error'][:60]}")

    # Save results
    results_path = THREAD_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
