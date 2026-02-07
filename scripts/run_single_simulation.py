"""
Run Single Thread Simulation

Wrapper script to run a complete simulation for one thread:
1. Runs the thread simulation using Mesa ABM
2. Runs validation comparing simulated vs real thread
3. Outputs summary results for aggregation

Usage:
    python scripts/run_single_simulation.py <thread_dir> [--output DIR] [--rounds N]

Example:
    python scripts/run_single_simulation.py batch_simulations/thread_001/ --output batch_output/thread_001/
"""

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for valid JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def run_simulation(config_path: str, output_dir: Path, max_rounds: int = 10) -> dict:
    """
    Run the thread simulation and return results.
    
    Returns dict with simulation_output path and metrics.
    """
    from sim.thread_simulation import ThreadModel
    
    print(f"\n{'='*60}")
    print("RUNNING THREAD SIMULATION")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Max rounds: {max_rounds}")
    
    # Create output directory
    sim_output = output_dir / 'simulation_output'
    sim_output.mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    model = ThreadModel(config_path=config_path)
    model.run(max_rounds=max_rounds)
    
    # Export results
    model.export_results(output_dir=str(sim_output))
    
    # Return metrics
    return {
        'total_posts': len(model.thread_history),
        'max_depth': max([p['depth'] for p in model.thread_history]),
        'rounds': model.current_round,
        'simulated_metadata_path': str(sim_output / 'simulated_thread_metadata.json'),
        'thread_history_path': str(sim_output / 'thread_history.json')
    }


def run_validation(real_path: str, simulated_path: str, output_dir: Path) -> dict:
    """
    Run validation comparing real vs simulated thread.

    Returns dict with validation metrics.
    """
    import gc
    import torch
    from scripts.validate_thread_simulation import ThreadValidator

    print(f"\n{'='*60}")
    print("RUNNING VALIDATION")
    print(f"{'='*60}")
    print(f"Real thread: {real_path}")
    print(f"Simulated thread: {simulated_path}")

    # Create validation output directory
    val_output = output_dir / 'validation_output'
    val_output.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = ThreadValidator(device='auto')
    
    # Load threads
    real_tweets = validator.load_thread(real_path)
    sim_tweets = validator.load_thread(simulated_path)
    
    print(f"  Real thread: {len(real_tweets)} tweets")
    print(f"  Simulated thread: {len(sim_tweets)} tweets")
    
    # Classify sentiment
    real_classified = validator.classify_sentiment(real_tweets)
    sim_classified = validator.classify_sentiment(sim_tweets)
    
    # Calculate distributions
    real_dist = validator.calculate_sentiment_distribution(real_classified)
    sim_dist = validator.calculate_sentiment_distribution(sim_classified)
    
    # Jensen-Shannon Divergence
    jsd = validator.calculate_jsd(real_dist, sim_dist)
    
    # Thread structure metrics
    real_depth = validator.calculate_max_depth(real_tweets)
    sim_depth = validator.calculate_max_depth(sim_tweets)
    
    real_engagement = validator.calculate_engagement_ratio(real_tweets)
    sim_engagement = validator.calculate_engagement_ratio(sim_tweets)
    
    # Calculate similarity scores
    sentiment_similarity = (1 - jsd) * 100
    depth_similarity = (1 - abs(real_depth - sim_depth) / max(real_depth, sim_depth, 1)) * 100
    engagement_similarity = (1 - abs(real_engagement - sim_engagement) / max(real_engagement, sim_engagement, 1)) * 100
    
    # Overall accuracy (weighted)
    overall_accuracy = (
        sentiment_similarity * 0.70 +
        depth_similarity * 0.15 +
        engagement_similarity * 0.15
    )
    
    # Visualization
    viz_path = val_output / 'sentiment_comparison.png'
    try:
        validator.visualize_comparison(real_dist, sim_dist, output_path=str(viz_path))
    except Exception as e:
        print(f"  Warning: Could not create visualization: {e}")
    
    # Compile results
    results = {
        'sentiment_distributions': {
            'real': real_dist,
            'simulated': sim_dist
        },
        'similarity_metrics': {
            'jensen_shannon_divergence': float(jsd),
            'sentiment_similarity_percent': float(sentiment_similarity),
            'depth_similarity_percent': float(depth_similarity),
            'engagement_similarity_percent': float(engagement_similarity),
            'overall_accuracy_percent': float(overall_accuracy)
        },
        'thread_structure': {
            'real': {
                'total_tweets': len(real_tweets),
                'max_depth': real_depth,
                'engagement_ratio': float(real_engagement)
            },
            'simulated': {
                'total_tweets': len(sim_tweets),
                'max_depth': sim_depth,
                'engagement_ratio': float(sim_engagement)
            }
        }
    }
    
    # Sanitize NaN/Inf before JSON serialization
    results = sanitize_for_json(results)

    # Save validation results
    results_path = val_output / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Validation complete!")
    print(f"  Sentiment similarity: {sentiment_similarity:.1f}%")
    print(f"  Overall accuracy: {overall_accuracy:.1f}%")

    # Free GPU memory from validation model
    del validator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def create_summary(thread_dir: Path, output_dir: Path, sim_results: dict, 
                   val_results: dict, config: dict, success: bool, error: str = None) -> dict:
    """Create summary.json for aggregation."""
    
    # Load thread info from config
    thread_info = config.get('thread_info', {})
    
    summary = {
        'thread_num': config.get('simulation', {}).get('thread_num', 0),
        'root_id': str(thread_info.get('root_tweet_id', '')),
        'root_text': thread_info.get('root_text', '')[:100],
        'success': success,
        'error': error,
        'timestamp': datetime.now().isoformat(),
        
        # Thread metadata
        'expected_replies': thread_info.get('internal_replies', 0),
        'expected_depth': thread_info.get('max_depth', 0),
        
        # Simulation results
        'simulation': sim_results if sim_results else None,
        
        # Validation results
        'validation': val_results if val_results else None
    }
    
    summary = sanitize_for_json(summary)

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run single thread simulation with validation')
    parser.add_argument('thread_dir', type=str,
                       help='Path to thread directory (e.g., batch_simulations/thread_001/)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: batch_output/<thread_name>/)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of simulation rounds (default: 10)')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step (useful if no real thread data)')
    
    args = parser.parse_args()
    
    # Resolve paths
    thread_dir = Path(args.thread_dir).resolve()
    
    if not thread_dir.exists():
        print(f"ERROR: Thread directory not found: {thread_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        # Default: batch_output/<thread_name>/
        project_root = Path(__file__).parent.parent
        thread_name = thread_dir.name
        output_dir = project_root / 'batch_output' / thread_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SINGLE THREAD SIMULATION RUNNER")
    print("="*80)
    print(f"\nThread directory: {thread_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load config
    config_path = thread_dir / 'config.yaml'
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize results
    sim_results = None
    val_results = None
    success = False
    error_msg = None
    
    try:
        # Step 1: Run simulation
        sim_results = run_simulation(
            config_path=str(config_path),
            output_dir=output_dir,
            max_rounds=args.rounds
        )
        
        # Step 2: Run validation (if we have real thread data)
        if not args.skip_validation:
            real_metadata_path = thread_dir / 'thread_metadata.json'
            simulated_metadata_path = output_dir / 'simulation_output' / 'simulated_thread_metadata.json'
            
            if real_metadata_path.exists() and simulated_metadata_path.exists():
                val_results = run_validation(
                    real_path=str(real_metadata_path),
                    simulated_path=str(simulated_metadata_path),
                    output_dir=output_dir
                )
            else:
                print(f"\nSkipping validation: Missing metadata files")
                if not real_metadata_path.exists():
                    print(f"  - Real: {real_metadata_path} NOT FOUND")
                if not simulated_metadata_path.exists():
                    print(f"  - Simulated: {simulated_metadata_path} NOT FOUND")
        
        success = True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\nERROR during simulation: {e}")
        traceback.print_exc()
    
    # Step 3: Create summary
    summary = create_summary(
        thread_dir=thread_dir,
        output_dir=output_dir,
        sim_results=sim_results,
        val_results=val_results,
        config=config,
        success=success,
        error=error_msg
    )
    
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success}")
    print(f"Output: {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
