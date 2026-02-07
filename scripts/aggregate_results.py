"""
Aggregate Batch Simulation Results

Combines results from all thread simulations into analysis-ready format.
Outputs:
- aggregated_results.csv: One row per thread with all metrics
- summary_statistics.json: Cross-thread statistics
- failed_simulations.txt: List of failed runs

Usage:
    python scripts/aggregate_results.py [--input DIR] [--output PREFIX]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def load_thread_summaries(batch_output_dir: Path) -> list:
    """Load all summary.json files from thread output directories."""
    summaries = []
    
    thread_dirs = sorted(batch_output_dir.glob('thread_*'))
    
    for thread_dir in thread_dirs:
        summary_path = thread_dir / 'summary.json'
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                    summary['_output_dir'] = str(thread_dir)
                    summaries.append(summary)
            except Exception as e:
                print(f"  Warning: Error reading {summary_path}: {e}")
        else:
            print(f"  Warning: No summary.json in {thread_dir}")
    
    return summaries


def extract_metrics(summary: dict) -> dict:
    """Extract all relevant metrics from a thread summary."""
    validation = summary.get('validation') or {}
    sim_metrics = validation.get('similarity_metrics', {})
    real_structure = validation.get('thread_structure', {}).get('real', {})
    sim_structure = validation.get('thread_structure', {}).get('simulated', {})
    real_sentiment = validation.get('sentiment_distributions', {}).get('real', {})
    sim_sentiment = validation.get('sentiment_distributions', {}).get('simulated', {})
    
    return {
        # Thread identification
        'thread_num': summary.get('thread_num', 0),
        'root_id': summary.get('root_id', ''),
        'root_text': summary.get('root_text', '')[:100],
        'success': summary.get('success', False),
        'error': summary.get('error', ''),
        'timestamp': summary.get('timestamp', ''),
        
        # Expected (from original data)
        'expected_replies': summary.get('expected_replies', 0),
        'expected_depth': summary.get('expected_depth', 0),
        
        # Real thread metrics
        'real_tweets': real_structure.get('total_tweets', 0),
        'real_depth': real_structure.get('max_depth', 0),
        'real_engagement': real_structure.get('engagement_ratio', 0),
        
        # Simulated thread metrics
        'sim_tweets': sim_structure.get('total_tweets', 0),
        'sim_depth': sim_structure.get('max_depth', 0),
        'sim_engagement': sim_structure.get('engagement_ratio', 0),
        
        # Real sentiment distribution
        'real_positive': real_sentiment.get('Positive', 0),
        'real_neutral': real_sentiment.get('Neutral', 0),
        'real_negative': real_sentiment.get('Negative', 0),
        
        # Simulated sentiment distribution
        'sim_positive': sim_sentiment.get('Positive', 0),
        'sim_neutral': sim_sentiment.get('Neutral', 0),
        'sim_negative': sim_sentiment.get('Negative', 0),
        
        # Similarity metrics
        'jensen_shannon_divergence': sim_metrics.get('jensen_shannon_divergence', None),
        'sentiment_similarity': sim_metrics.get('sentiment_similarity_percent', None),
        'depth_similarity': sim_metrics.get('depth_similarity_percent', None),
        'engagement_similarity': sim_metrics.get('engagement_similarity_percent', None),
        'overall_accuracy': sim_metrics.get('overall_accuracy_percent', None),
    }


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate summary statistics across all threads."""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_threads': len(df),
        'successful': int(df['success'].sum()),
        'failed': int((~df['success']).sum()),
        'success_rate': float(df['success'].mean() * 100)
    }
    
    # Filter to successful runs with validation data
    valid_df = df[(df['success']) & (df['overall_accuracy'].notna())]
    
    if len(valid_df) > 0:
        # Sentiment similarity stats
        stats['sentiment_similarity'] = {
            'count': int(len(valid_df)),
            'mean': float(valid_df['sentiment_similarity'].mean()),
            'std': float(valid_df['sentiment_similarity'].std()),
            'min': float(valid_df['sentiment_similarity'].min()),
            'max': float(valid_df['sentiment_similarity'].max()),
            'median': float(valid_df['sentiment_similarity'].median())
        }
        
        # Overall accuracy stats
        stats['overall_accuracy'] = {
            'count': int(len(valid_df)),
            'mean': float(valid_df['overall_accuracy'].mean()),
            'std': float(valid_df['overall_accuracy'].std()),
            'min': float(valid_df['overall_accuracy'].min()),
            'max': float(valid_df['overall_accuracy'].max()),
            'median': float(valid_df['overall_accuracy'].median())
        }
        
        # JSD stats
        stats['jensen_shannon_divergence'] = {
            'count': int(len(valid_df)),
            'mean': float(valid_df['jensen_shannon_divergence'].mean()),
            'std': float(valid_df['jensen_shannon_divergence'].std()),
            'min': float(valid_df['jensen_shannon_divergence'].min()),
            'max': float(valid_df['jensen_shannon_divergence'].max()),
            'median': float(valid_df['jensen_shannon_divergence'].median())
        }
        
        # Sentiment distribution averages
        stats['avg_real_sentiment'] = {
            'positive': float(valid_df['real_positive'].mean()),
            'neutral': float(valid_df['real_neutral'].mean()),
            'negative': float(valid_df['real_negative'].mean())
        }
        
        stats['avg_sim_sentiment'] = {
            'positive': float(valid_df['sim_positive'].mean()),
            'neutral': float(valid_df['sim_neutral'].mean()),
            'negative': float(valid_df['sim_negative'].mean())
        }
        
        # Thread structure averages
        stats['avg_structure'] = {
            'real_tweets': float(valid_df['real_tweets'].mean()),
            'sim_tweets': float(valid_df['sim_tweets'].mean()),
            'real_depth': float(valid_df['real_depth'].mean()),
            'sim_depth': float(valid_df['sim_depth'].mean())
        }
    else:
        stats['sentiment_similarity'] = None
        stats['overall_accuracy'] = None
        stats['jensen_shannon_divergence'] = None
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Aggregate batch simulation results')
    parser.add_argument('--input', type=str, default='batch_output',
                       help='Directory containing thread output directories')
    parser.add_argument('--output', type=str, default='batch_output/aggregated',
                       help='Output prefix for aggregated files')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / args.input
    output_prefix = project_root / args.output
    
    print("="*80)
    print("AGGREGATE BATCH SIMULATION RESULTS")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output prefix: {output_prefix}")
    
    # Create output directory
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # Load summaries
    print("\nLoading thread summaries...")
    summaries = load_thread_summaries(input_dir)
    print(f"Found {len(summaries)} thread summaries")
    
    if not summaries:
        print("\nNo summaries found. Run simulations first.")
        sys.exit(1)
    
    # Extract metrics
    print("\nExtracting metrics...")
    metrics_list = [extract_metrics(s) for s in summaries]
    df = pd.DataFrame(metrics_list)
    
    # Sort by thread number
    df = df.sort_values('thread_num')
    
    # Save aggregated CSV
    csv_path = Path(str(output_prefix) + '_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    
    # Calculate and save statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(df)
    
    stats_path = Path(str(output_prefix) + '_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved: {stats_path}")
    
    # Save failed simulations list
    failed_df = df[~df['success']]
    if len(failed_df) > 0:
        failed_path = Path(str(output_prefix) + '_failed.txt')
        with open(failed_path, 'w') as f:
            f.write("FAILED SIMULATIONS\n")
            f.write("="*60 + "\n\n")
            for _, row in failed_df.iterrows():
                f.write(f"Thread {row['thread_num']}: {row['root_id']}\n")
                f.write(f"  Error: {row['error']}\n\n")
        print(f"✓ Saved: {failed_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal threads: {stats['total_threads']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    if stats.get('overall_accuracy'):
        print(f"\nOverall Accuracy (successful runs):")
        print(f"  Mean: {stats['overall_accuracy']['mean']:.1f}%")
        print(f"  Std:  {stats['overall_accuracy']['std']:.1f}%")
        print(f"  Min:  {stats['overall_accuracy']['min']:.1f}%")
        print(f"  Max:  {stats['overall_accuracy']['max']:.1f}%")
        
        print(f"\nSentiment Similarity:")
        print(f"  Mean: {stats['sentiment_similarity']['mean']:.1f}%")
        print(f"  Std:  {stats['sentiment_similarity']['std']:.1f}%")
    
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {stats_path}")
    if len(failed_df) > 0:
        print(f"  - {failed_path}")


if __name__ == '__main__':
    main()
