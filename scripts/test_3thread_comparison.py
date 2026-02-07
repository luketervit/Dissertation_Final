"""
Test 3 threads with 2 models (dolphin-llama3:8b vs llama3.1:8b).

Uses REAL thread participants from batch_simulations/ directories.
For each thread, re-generates agents_for_thread.csv by matching user_ids
from temporal_events to the global classified agents pool.

Usage (on GCP):
    python scripts/test_3thread_comparison.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = ["dolphin-llama3:8b", "llama3.1:8b"]
BATCH_DIR = Path("batch_simulations")
AGENTS_CSV = Path("processed_agents/processed_agents_august.csv")
TEST_THREADS = [1, 2, 3]  # thread_001, thread_002, thread_003
MAX_ROUNDS = 5


def rebuild_agents_for_thread(thread_dir: Path, global_agents: pd.DataFrame) -> int:
    """Rebuild agents_for_thread.csv using REAL thread participants only.

    Reads temporal_events from thread_metadata.json, extracts unique user_ids,
    and matches them to the global classified agents pool.

    Returns the number of matched agents.
    """
    metadata_path = thread_dir / "thread_metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: No thread_metadata.json in {thread_dir}")
        return 0

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Extract unique user_ids from temporal_events
    events = metadata.get("temporal_events", [])
    if not events:
        print(f"  WARNING: No temporal_events in {thread_dir}")
        return 0

    user_ids = set()
    for event in events:
        uid = event.get("user_id")
        if uid and str(uid) not in ("None", "nan", "unknown"):
            user_ids.add(str(uid))

    if not user_ids:
        print(f"  WARNING: No valid user_ids in temporal_events for {thread_dir}")
        return 0

    # Match to global agents pool
    global_agents["user_id_str"] = global_agents["user_id"].astype(str)
    matched = global_agents[global_agents["user_id_str"].isin(user_ids)].copy()
    matched = matched.drop(columns=["user_id_str"])

    if len(matched) == 0:
        print(f"  WARNING: No agents matched from {len(user_ids)} user_ids in {thread_dir}")
        return 0

    # Overwrite agents_for_thread.csv with real participants
    matched.to_csv(thread_dir / "agents_for_thread.csv", index=False)
    print(f"  Rebuilt agents: {len(matched)}/{len(user_ids)} users matched "
          f"({len(user_ids) - len(matched)} missing from global pool)")
    return len(matched)


def run_simulation(thread_dir: Path, model: str, output_dir: Path) -> dict:
    """Run one simulation and return basic metrics."""
    from sim.thread_simulation import ThreadModel

    # Patch config with the target model and updated settings
    config_path = thread_dir / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["llm"]["model"] = model
    config["llm"]["temperature"] = 0.9
    # Remove old system_prompt if present (now built per-agent in code)
    config["llm"].pop("system_prompt", None)

    # Write patched config to output dir
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
        "num_agents": len(model_instance.agent_list),
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
    print("3-THREAD MODEL COMPARISON TEST (Real Participants)")
    print("=" * 80)

    # Load global agents pool
    if not AGENTS_CSV.exists():
        print(f"ERROR: {AGENTS_CSV} not found. Run gcp_full_pipeline.py step 3 first.")
        sys.exit(1)

    global_agents = pd.read_csv(AGENTS_CSV)
    print(f"Loaded {len(global_agents)} agents from {AGENTS_CSV}")

    # Find thread directories
    thread_dirs = []
    for num in TEST_THREADS:
        d = BATCH_DIR / f"thread_{num:03d}"
        if d.exists():
            thread_dirs.append(d)
        else:
            print(f"WARNING: {d} does not exist, skipping")

    if not thread_dirs:
        print("ERROR: No thread directories found!")
        sys.exit(1)

    # Rebuild agents_for_thread.csv with real participants
    print("\n--- Rebuilding agent files with real thread participants ---")
    for td in thread_dirs:
        print(f"\n{td.name}:")
        rebuild_agents_for_thread(td, global_agents)

    print(f"\nModels: {MODELS}")
    print(f"Threads: {[t.name for t in thread_dirs]}")
    print(f"Rounds: {MAX_ROUNDS}")
    print()

    all_results = []

    for model in MODELS:
        print(f"\n{'#' * 80}")
        print(f"MODEL: {model}")
        print(f"{'#' * 80}")

        for thread_dir in thread_dirs:
            output_dir = BATCH_DIR / f"test_results_{model.replace(':', '_')}" / thread_dir.name
            print(f"\n--- {thread_dir.name} with {model} ---")

            try:
                result = run_simulation(thread_dir, model, output_dir)
                all_results.append(result)
                print(f"  Agents: {result['num_agents']}, "
                      f"Posts: {result['total_posts']}, "
                      f"Depth: {result['max_depth']}, "
                      f"Time: {result['elapsed_s']}s")
                print(f"  Left: {result['left_pct']:.0f}%, "
                      f"Right: {result['right_pct']:.0f}%, "
                      f"Mean agg: {result['mean_aggression']:.3f}")
                print(f"  Sample tweets:")
                for t in result["sample_tweets"][:3]:
                    print(f"    \"{t[:120]}\"")
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

    for thread_dir in thread_dirs:
        print(f"\n{thread_dir.name}:")
        for model in MODELS:
            r = next(
                (x for x in all_results
                 if x.get("thread") == thread_dir.name and x.get("model") == model),
                None,
            )
            if r and "error" not in r:
                print(
                    f"  {model:25s} | Agents: {r['num_agents']:3d} | "
                    f"Posts: {r['total_posts']:3d} | "
                    f"L/R: {r['left_pct']:4.0f}/{r['right_pct']:4.0f}% | "
                    f"Agg: {r['mean_aggression']:.3f} | "
                    f"Time: {r['elapsed_s']}s"
                )
            elif r:
                print(f"  {model:25s} | ERROR: {r['error'][:60]}")

    # Save results
    results_path = BATCH_DIR / "test_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
