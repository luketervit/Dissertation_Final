"""
GCP Batch Simulation Runner

Orchestrates running multiple thread simulations on a GCP VM.
Supports parallel execution using Python multiprocessing.
Includes checkpointing for resume on failure.

Usage:
    # Run all threads sequentially
    python scripts/gcp_batch_runner.py --mode sequential

    # Run threads in parallel (recommended for GCP)
    python scripts/gcp_batch_runner.py --mode parallel --workers 4
    
    # Run specific range of threads
    python scripts/gcp_batch_runner.py --start 1 --end 20 --mode parallel

    # Resume from checkpoint (auto-skips completed threads)
    python scripts/gcp_batch_runner.py --resume

    # Dry run (prepare without executing)
    python scripts/gcp_batch_runner.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


CHECKPOINT_FILE = 'checkpoint.json'


def load_checkpoint(output_dir: Path) -> dict:
    """Load checkpoint file if it exists."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(output_dir: Path, completed: list, results: list):
    """Save checkpoint with completed threads and results."""
    checkpoint_path = output_dir / CHECKPOINT_FILE
    checkpoint = {
        'completed': completed,
        'results': results,
        'last_updated': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def get_thread_dirs(batch_dir: Path, start: int = 1, end: int = None) -> list:
    """Get list of thread directories to process."""
    all_dirs = sorted(batch_dir.glob('thread_*'))
    
    # Filter by range
    filtered = []
    for d in all_dirs:
        try:
            num = int(d.name.split('_')[1])
            if num >= start and (end is None or num <= end):
                filtered.append(d)
        except:
            continue
    
    return filtered


def run_single_thread(args: tuple) -> dict:
    """
    Run simulation for a single thread.
    
    Args: tuple of (thread_dir, output_dir, rounds, python_path, skip_validation)
    Returns: dict with thread results
    """
    thread_dir, output_base, rounds, python_path, skip_validation = args
    
    thread_name = thread_dir.name
    output_dir = output_base / thread_name
    
    start_time = time.time()
    
    try:
        # Run the single simulation script
        cmd = [
            python_path,
            'scripts/run_single_simulation.py',
            str(thread_dir),
            '--output', str(output_dir),
            '--rounds', str(rounds)
        ]
        if skip_validation:
            cmd.append('--skip-validation')
        
        result = subprocess.run(
            cmd,
            cwd=str(thread_dir.parent.parent),  # Project root
            capture_output=False, # Let stdout/stderr show in console for live monitoring
            text=True,
            timeout=None  # No timeout - let threads run as long as needed
        )
        
        elapsed = time.time() - start_time
        
        # Check if summary exists
        summary_path = output_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            return {
                'thread': thread_name,
                'success': summary.get('success', False),
                'elapsed_seconds': elapsed,
                'root_id': summary.get('root_id', ''),
                'validation': (summary.get('validation') or {}).get('similarity_metrics', {}),
                'error': summary.get('error')
            }
        else:
            return {
                'thread': thread_name,
                'success': False,
                'elapsed_seconds': elapsed,
                'error': "No summary.json created. Check console logs for error details."
            }
            
    except subprocess.TimeoutExpired:
        return {
            'thread': thread_name,
            'success': False,
            'elapsed_seconds': 1800,
            'error': 'Timeout after 30 minutes'
        }
    except Exception as e:
        return {
            'thread': thread_name,
            'success': False,
            'elapsed_seconds': 0,
            'error': str(e)
        }


def run_sequential(thread_dirs: list, output_dir: Path, rounds: int,
                   python_path: str, checkpoint: dict,
                   skip_validation: bool = False) -> list:
    """Run threads one at a time with checkpointing."""
    results = list(checkpoint.get('results', []))
    completed = set(checkpoint.get('completed', []))
    total = len(thread_dirs)

    for i, thread_dir in enumerate(thread_dirs, 1):
        thread_name = thread_dir.name

        # Skip if already completed
        if thread_name in completed:
            print(f"[{i}/{total}] ⏭ {thread_name} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing thread {i}/{total}: {thread_name}")
        print(f"{'='*60}")

        result = run_single_thread((thread_dir, output_dir, rounds, python_path, skip_validation))
        results.append(result)
        completed.add(thread_name)
        
        # Save checkpoint after each thread
        save_checkpoint(output_dir, list(completed), results)
        
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"\n{status} - {result['thread']} ({result['elapsed_seconds']:.1f}s)")
        if result.get('error'):
            print(f"  Error: {result['error'][:200]}")
        
        print(f"  [Checkpoint saved: {len(completed)}/{total} complete]")
    
    return results


def run_parallel(thread_dirs: list, output_dir: Path, rounds: int,
                 python_path: str, max_workers: int, checkpoint: dict,
                 skip_validation: bool = False) -> list:
    """Run threads in parallel with checkpointing."""
    results = list(checkpoint.get('results', []))
    completed = set(checkpoint.get('completed', []))
    
    # Filter out already completed threads
    pending_dirs = [td for td in thread_dirs if td.name not in completed]
    total = len(thread_dirs)
    skipped = len(thread_dirs) - len(pending_dirs)
    
    if skipped > 0:
        print(f"\n⏭ Skipping {skipped} already-completed threads")
    
    if not pending_dirs:
        print("All threads already completed!")
        return results
    
    # Prepare arguments
    args_list = [(td, output_dir, rounds, python_path, skip_validation) for td in pending_dirs]
    
    print(f"\nStarting parallel execution with {max_workers} workers...")
    print(f"Processing {len(pending_dirs)} remaining threads\n")
    
    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_thread, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            thread_dir = futures[future]
            processed += 1
            
            try:
                result = future.result()
                results.append(result)
                completed.add(result['thread'])
                
                # Save checkpoint after each completion
                save_checkpoint(output_dir, list(completed), results)
                
                status = "✓" if result['success'] else "✗"
                print(f"[{skipped + processed}/{total}] {status} {result['thread']} ({result['elapsed_seconds']:.1f}s) [checkpoint saved]")
                
            except Exception as e:
                thread_name = thread_dir.name
                results.append({
                    'thread': thread_name,
                    'success': False,
                    'error': str(e)
                })
                completed.add(thread_name)
                save_checkpoint(output_dir, list(completed), results)
                print(f"[{skipped + processed}/{total}] ✗ {thread_name} - Exception: {e}")
    
    return results


def save_batch_results(results: list, output_dir: Path, start_time: datetime) -> Path:
    """Save consolidated batch results."""
    end_time = datetime.now()
    
    # Calculate summary stats
    total = len(results)
    successful = sum(1 for r in results if r.get('success'))
    failed = total - successful
    
    # Get accuracy stats for successful runs
    accuracies = []
    for r in results:
        if r.get('success') and r.get('validation'):
            acc = r['validation'].get('overall_accuracy_percent')
            if acc is not None:
                accuracies.append(acc)
    
    batch_summary = {
        'batch_info': {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'total_threads': total,
            'successful': successful,
            'failed': failed,
            'success_rate_percent': (successful / total * 100) if total > 0 else 0
        },
        'accuracy_stats': {
            'count': len(accuracies),
            'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else None,
            'min_accuracy': min(accuracies) if accuracies else None,
            'max_accuracy': max(accuracies) if accuracies else None
        },
        'thread_results': results
    }
    
    # Save batch results
    batch_results_path = output_dir / 'batch_results.json'
    with open(batch_results_path, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    return batch_results_path


def main():
    parser = argparse.ArgumentParser(description='Run batch thread simulations on GCP')
    parser.add_argument('--batch-dir', type=str, default='batch_simulations',
                       help='Directory containing thread simulation configs')
    parser.add_argument('--output', type=str, default='batch_output',
                       help='Output directory for results')
    parser.add_argument('--start', type=int, default=1,
                       help='Starting thread number (1-indexed)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending thread number (inclusive)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Simulation rounds per thread')
    parser.add_argument('--mode', type=str, choices=['sequential', 'parallel'], default='sequential',
                       help='Execution mode')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (for parallel mode)')
    parser.add_argument('--python', type=str, default=sys.executable,
                       help='Python interpreter path')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint (auto-enabled if checkpoint exists)')
    parser.add_argument('--fresh', action='store_true',
                       help='Ignore checkpoint and start fresh')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step (useful if torch not installed)')

    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    batch_dir = project_root / args.batch_dir
    output_dir = project_root / args.output
    
    print("="*80)
    print("GCP BATCH SIMULATION RUNNER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch directory: {batch_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Thread range: {args.start} to {args.end or 'end'}")
    print(f"  Rounds per thread: {args.rounds}")
    print(f"  Execution mode: {args.mode}")
    if args.mode == 'parallel':
        print(f"  Workers: {args.workers}")
    print(f"  Python: {args.python}")
    
    # Check batch directory exists
    if not batch_dir.exists():
        print(f"\nERROR: Batch directory not found: {batch_dir}")
        print("Run prepare_batch_simulations.py first.")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = {} if args.fresh else load_checkpoint(output_dir)
    if checkpoint.get('completed'):
        print(f"\n📁 Found checkpoint: {len(checkpoint['completed'])} threads already completed")
        if not args.resume and not args.fresh:
            print("  Use --resume to continue or --fresh to restart")
            args.resume = True  # Auto-resume if checkpoint exists
    
    # Get thread directories
    thread_dirs = get_thread_dirs(batch_dir, args.start, args.end)
    
    if not thread_dirs:
        print(f"\nNo thread directories found in range {args.start}-{args.end}")
        sys.exit(1)
    
    # Show pending vs completed
    completed_set = set(checkpoint.get('completed', []))
    pending = [td for td in thread_dirs if td.name not in completed_set]
    
    print(f"\nThreads: {len(thread_dirs)} total, {len(pending)} pending, {len(completed_set)} completed")
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(pending)} threads")
        sys.exit(0)
    
    # Run simulations
    start_time = datetime.now()
    print(f"\nStarting batch execution at {start_time.isoformat()}")
    print("-"*80)
    
    if args.mode == 'sequential':
        results = run_sequential(thread_dirs, output_dir, args.rounds, args.python, checkpoint, args.skip_validation)
    else:
        results = run_parallel(thread_dirs, output_dir, args.rounds, args.python, args.workers, checkpoint, args.skip_validation)
    
    # Save results
    print("\n" + "-"*80)
    print("Saving batch results...")
    
    results_path = save_batch_results(results, output_dir, start_time)
    
    # Final summary
    successful = sum(1 for r in results if r.get('success'))
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print("BATCH EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal threads: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_path}")
    
    if failed > 0:
        print(f"\nFailed threads:")
        for r in results:
            if not r.get('success'):
                print(f"  - {r['thread']}: {r.get('error', 'Unknown error')[:80]}")


if __name__ == '__main__':
    main()
