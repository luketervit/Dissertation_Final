"""
Test 3 threads with 2 models (dolphin-llama3:8b vs llama3.1:8b).

Uses REAL thread participants by scanning raw data chunks for each thread's
conversationId, extracting user_ids, and matching to the classified agents pool.

Usage (on GCP):
    python scripts/test_3thread_comparison.py
"""
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = ["dolphin-llama3:8b", "llama3.1:8b"]
BATCH_DIR = Path("batch_simulations")
AGENTS_DIR = Path("processed_agents")
DATA_DIR = Path("data")
TEST_THREADS = [1, 2, 3]  # thread_001, thread_002, thread_003
MAX_ROUNDS = 5


def find_thread_user_ids(conversation_id: str) -> set[str]:
    """Scan raw data chunks to find all user_ids who participated in a thread."""
    chunks = sorted(DATA_DIR.glob("aug_chunk_*.csv"))
    if not chunks:
        print(f"  WARNING: No aug_chunk_*.csv files in {DATA_DIR}")
        return set()

    user_ids = set()
    for chunk_file in chunks:
        try:
            df = pd.read_csv(chunk_file, dtype={
                "id": str, "conversationId": str,
            }, usecols=lambda c: c in ("id", "conversationId", "user", "userId"))

            matched = df[df["conversationId"] == conversation_id]
            if len(matched) == 0:
                continue

            for _, row in matched.iterrows():
                uid = None
                if "userId" in row and pd.notna(row.get("userId")):
                    uid = str(row["userId"])
                elif "user" in row and pd.notna(row.get("user")):
                    m = re.search(r"'id':\s*(\d+)", str(row["user"]))
                    uid = m.group(1) if m else None

                if uid and uid not in ("None", "nan"):
                    user_ids.add(uid)
        except Exception:
            pass

    return user_ids


def rebuild_agents_for_thread(
    thread_dir: Path, global_agents: pd.DataFrame, global_agents_str: pd.Series
) -> int:
    """Rebuild agents_for_thread.csv using REAL thread participants only.

    Scans raw data to find user_ids by conversationId, then matches to
    the global classified agents pool.

    Returns the number of matched agents.
    """
    # Get the thread's conversationId from config or metadata
    config_path = thread_dir / "config.yaml"
    metadata_path = thread_dir / "thread_metadata.json"

    conversation_id = None
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        conversation_id = str(config.get("target_tweet_id", ""))

    if not conversation_id and metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        conversation_id = str(metadata.get("root_tweet", {}).get("id", ""))

    if not conversation_id:
        print(f"  WARNING: Could not determine conversationId for {thread_dir}")
        return 0

    print(f"  Scanning raw data for conversationId={conversation_id}...")
    user_ids = find_thread_user_ids(conversation_id)

    if not user_ids:
        print(f"  WARNING: No users found in raw data for thread {conversation_id}")
        return 0

    # Match to global agents pool
    matched = global_agents[global_agents_str.isin(user_ids)].copy()

    if len(matched) == 0:
        print(f"  WARNING: Found {len(user_ids)} users but none matched agents pool")
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

    # Load global agents pool (supports single file or chunked files)
    single_file = AGENTS_DIR / "processed_agents_august.csv"
    chunk_files = sorted(AGENTS_DIR.glob("processed_agents_chunk_*.csv"))

    if single_file.exists():
        global_agents = pd.read_csv(single_file)
        print(f"Loaded {len(global_agents)} agents from {single_file}")
    elif chunk_files:
        print(f"Loading agents from {len(chunk_files)} chunk files...")
        global_agents = pd.concat(
            [pd.read_csv(f) for f in chunk_files], ignore_index=True
        )
        # Deduplicate by user_id (same user may appear in multiple chunks)
        global_agents = global_agents.drop_duplicates(subset=["user_id"], keep="first")
        print(f"Loaded {len(global_agents)} unique agents from {len(chunk_files)} chunks")
    else:
        print(f"ERROR: No agent files found in {AGENTS_DIR}/")
        print(f"  Expected: processed_agents_august.csv or processed_agents_chunk_*.csv")
        sys.exit(1)

    # Pre-compute string user_ids for matching
    global_agents_str = global_agents["user_id"].astype(str)

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
        rebuild_agents_for_thread(td, global_agents, global_agents_str)

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
