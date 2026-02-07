"""
Run Batch Simulations

Iterates over prepared thread directories and runs thread_simulation.py
on each one. Designed for GCP with Ollama running locally.

Saves output to batch_output_100/thread_NNN/simulation_output/

Usage:
    python scripts/run_batch_simulations.py [--input batch_simulations_100] [--output batch_output_100]

    # Run specific range (for parallel workers):
    python scripts/run_batch_simulations.py --start 1 --end 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_single_thread(thread_dir: Path, output_dir: Path) -> dict:
    """
    Run simulation for a single thread.
    Returns dict with status and timing info.
    """
    from sim.thread_simulation import ThreadModel

    config_path = thread_dir / 'config.yaml'
    if not config_path.exists():
        return {'status': 'skipped', 'reason': 'no config.yaml'}

    sim_output = output_dir / 'simulation_output'
    sim_output.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    try:
        model = ThreadModel(config_path=str(config_path))

        # Get max_rounds from config
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        max_rounds = config.get('simulation', {}).get('max_rounds', 10)

        model.run(max_rounds=max_rounds)
        model.export_results(output_dir=str(sim_output))

        elapsed = time.time() - start_time
        total_posts = len(model.thread_history)

        return {
            'status': 'success',
            'total_posts': total_posts,
            'elapsed_seconds': round(elapsed, 1),
            'agents': len(model.agent_list),
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'elapsed_seconds': round(elapsed, 1),
        }


def main():
    parser = argparse.ArgumentParser(description='Run batch simulations')
    parser.add_argument('--input', type=str, default='batch_simulations_100',
                        help='Input directory with prepared threads')
    parser.add_argument('--output', type=str, default='batch_output_100',
                        help='Output directory for simulation results')
    parser.add_argument('--start', type=int, default=1, help='Start thread number')
    parser.add_argument('--end', type=int, default=999, help='End thread number')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_dir = project_root / args.output

    # Find all thread directories
    thread_dirs = sorted(input_dir.glob('thread_*'))
    if not thread_dirs:
        print(f"ERROR: No thread directories found in {input_dir}")
        sys.exit(1)

    # Filter by range
    thread_dirs = [
        d for d in thread_dirs
        if args.start <= int(d.name.split('_')[1]) <= args.end
    ]

    print("=" * 80)
    print(f"BATCH SIMULATION: {len(thread_dirs)} threads")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Range: {args.start}-{args.end}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total_start = time.time()

    for i, thread_dir in enumerate(thread_dirs):
        thread_name = thread_dir.name
        thread_output = output_dir / thread_name

        # Skip if already completed
        done_marker = thread_output / 'simulation_output' / 'simulated_thread_metadata.json'
        if done_marker.exists():
            print(f"\n[{i+1}/{len(thread_dirs)}] {thread_name}: ALREADY DONE, skipping")
            results.append({'thread': thread_name, 'status': 'skipped_done'})
            continue

        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(thread_dirs)}] Running {thread_name}...")

        result = run_single_thread(thread_dir, thread_output)
        result['thread'] = thread_name
        results.append(result)

        if result['status'] == 'success':
            print(f"  OK: {result['total_posts']} posts, {result['agents']} agents, "
                  f"{result['elapsed_seconds']}s")
        else:
            print(f"  FAILED: {result.get('error', 'unknown')}")

        # Save progress after each thread
        progress = {
            'completed': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'skipped': sum(1 for r in results if 'skipped' in r['status']),
            'total_elapsed': round(time.time() - total_start, 1),
            'results': results,
        }
        with open(output_dir / 'batch_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

    # Final summary
    total_elapsed = time.time() - total_start
    successes = sum(1 for r in results if r['status'] == 'success')
    failures = sum(1 for r in results if r['status'] == 'error')

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"  Success: {successes}/{len(thread_dirs)}")
    print(f"  Failed: {failures}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    if successes > 0:
        avg_time = sum(r['elapsed_seconds'] for r in results if r['status'] == 'success') / successes
        print(f"  Avg time per thread: {avg_time:.1f}s")


if __name__ == '__main__':
    main()
