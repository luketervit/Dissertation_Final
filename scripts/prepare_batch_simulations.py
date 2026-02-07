"""
Prepare Batch Simulations (Optimized)

Reads the top threads from august_thread_depths.csv and prepares individual
simulation directories for each thread. Uses data already in the depths file
to avoid expensive scanning of raw data chunks.

Each directory contains:
- config.yaml: Thread-specific configuration
- thread_metadata.json: Root tweet and thread info
- agents_for_thread.csv: Agent DNA profiles (from processed_agents)

Usage:
    python scripts/prepare_batch_simulations.py [--limit N] [--output DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_thread_depths(depths_path: str, limit: int = 100) -> pd.DataFrame:
    """Load and return top N threads from august_thread_depths.csv."""
    print(f"Loading thread depths from {depths_path}...")
    df = pd.read_csv(depths_path)
    actual_count = min(limit, len(df))
    print(f"  Found {len(df)} threads, using top {actual_count}")
    return df.head(limit)


def load_all_agents(agents_dir: str) -> pd.DataFrame:
    """
    Load all agent profiles at once for faster lookup.
    Checks for both chunk files and august-specific file.
    """
    agents_path = Path(agents_dir)
    
    # First check for august-specific processed agents
    august_file = agents_path / 'processed_agents_august.csv'
    if august_file.exists():
        print(f"Loading agent profiles from {august_file.name}...")
        df = pd.read_csv(august_file, dtype={'user_id': str})
        print(f"  Loaded {len(df)} agent profiles")
        return df
    
    # Fall back to chunk files
    agent_files = sorted(agents_path.glob('processed_agents_chunk_*.csv'))
    
    print(f"Loading agent profiles from {len(agent_files)} files...")
    
    all_agents = []
    for agent_file in agent_files:
        try:
            df = pd.read_csv(agent_file, dtype={'user_id': str})
            all_agents.append(df)
        except Exception as e:
            print(f"  Warning: Error reading {agent_file.name}: {e}")
    
    if not all_agents:
        print("  WARNING: No agent files found!")
        return pd.DataFrame()
    
    combined = pd.concat(all_agents, ignore_index=True).drop_duplicates(subset=['user_id'])
    print(f"  Loaded {len(combined)} unique agent profiles")
    return combined


def sample_agents_for_thread(all_agents: pd.DataFrame, target_count: int, 
                             seed: int = None) -> pd.DataFrame:
    """
    Sample agents for a thread simulation.
    Uses stratified sampling to maintain political distribution.
    """
    if len(all_agents) == 0:
        return pd.DataFrame()
    
    # Cap at available agents
    actual_count = min(target_count, len(all_agents))
    
    # Try to maintain political distribution
    if 'political_label' in all_agents.columns:
        # Get distribution
        dist = all_agents['political_label'].value_counts(normalize=True)
        
        sampled_parts = []
        remaining = actual_count
        
        for label in ['Left', 'Right', 'Center']:
            if label in dist:
                n = int(dist[label] * actual_count)
                label_agents = all_agents[all_agents['political_label'] == label]
                n = min(n, len(label_agents), remaining)
                if n > 0:
                    sampled_parts.append(label_agents.sample(n=n, random_state=seed))
                    remaining -= n
        
        if sampled_parts:
            sampled = pd.concat(sampled_parts, ignore_index=True)
            # Fill any remaining with random sample
            if len(sampled) < actual_count:
                remaining_agents = all_agents[~all_agents['user_id'].isin(sampled['user_id'])]
                extra_count = min(actual_count - len(sampled), len(remaining_agents))
                if extra_count > 0:
                    extra = remaining_agents.sample(n=extra_count, random_state=seed)
                    sampled = pd.concat([sampled, extra], ignore_index=True)
            return sampled
    
    # Fallback: simple random sample
    return all_agents.sample(n=actual_count, random_state=seed)


def create_thread_config(thread_dir: Path, thread_info: dict, thread_num: int) -> None:
    """Create thread-specific config.yaml."""
    config = {
        'target_tweet_id': str(thread_info['root_id']),
        'paths': {
            'thread_metadata': str(thread_dir / 'thread_metadata.json'),
            'agents_for_thread': str(thread_dir / 'agents_for_thread.csv'),
        },
        'abm': {
            'lurker_ratio': 0,
            'lurker_dist_strategy': 'match_active_agents',
            'bounded_confidence_threshold': 0.3,
            'backfire_threshold': 0.6,
            'backfire_aggression_min': 0.7
        },
        'llm': {
            'provider': 'ollama',
            'model': 'dolphin-llama3:8b',
            'api_key_env': None,
            'temperature': 0.9,
            'max_tokens': 150,
        },
        'thread_info': {
            'root_tweet_id': str(thread_info['root_id']),
            'internal_replies': int(thread_info['internal_replies']),
            'total_tweets': int(thread_info['total_tweets']),
            'max_depth': int(thread_info['max_depth']),
            'root_text': str(thread_info['text'])[:500]
        },
        'simulation': {
            'max_rounds': 10,
            'thread_num': thread_num
        }
    }
    
    config_path = thread_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_thread_metadata(thread_dir: Path, thread_info: dict) -> None:
    """Create thread_metadata.json matching expected format."""
    root_id = str(thread_info['root_id'])
    
    metadata = {
        'root_tweet': {
            'id': root_id,
            'conversation_id': root_id,
            'user_id': 'unknown',
            'text': str(thread_info['text']),
            'expected_replies': int(thread_info['internal_replies']),
        },
        'replies': {
            'actual_count': int(thread_info['internal_replies']),
            'tweet_ids': [],
        },
        'thread_structure': {
            'max_depth': int(thread_info['max_depth']),
            'depth_distribution': str(thread_info['depth_dist'])
        },
        'source': 'august_thread_depths'
    }
    
    metadata_path = thread_dir / 'thread_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare batch simulations from august_thread_depths')
    parser.add_argument('--limit', type=int, default=100,
                       help='Maximum number of threads to prepare (default: 100)')
    parser.add_argument('--output', type=str, default='batch_simulations',
                       help='Output directory for simulation configs')
    parser.add_argument('--depths-file', type=str, default='output/august_thread_depths.csv',
                       help='Path to august_thread_depths.csv')
    parser.add_argument('--agents-dir', type=str, default='processed_agents',
                       help='Directory containing processed_agents_chunk_*.csv files')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve paths
    depths_path = project_root / args.depths_file
    output_dir = project_root / args.output
    agents_dir = project_root / args.agents_dir
    
    print("="*80)
    print("PREPARE BATCH SIMULATIONS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Thread depths file: {depths_path}")
    print(f"  Agents directory: {agents_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max threads: {args.limit}")
    
    # Load thread depths
    print("\n" + "-"*80)
    threads_df = load_thread_depths(str(depths_path), args.limit)
    actual_count = len(threads_df)
    
    # Load all agents once (much faster than per-thread)
    print("\n" + "-"*80)
    all_agents = load_all_agents(str(agents_dir))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each thread
    print("\n" + "-"*80)
    print("Creating thread simulation directories...")
    
    for idx, row in threads_df.iterrows():
        thread_info = row.to_dict()
        thread_num = idx + 1
        thread_dir = output_dir / f'thread_{thread_num:03d}'
        
        # Use internal_replies as target agent count (matches actual thread size)
        target_agents = int(thread_info['internal_replies'])
        
        print(f"  Thread {thread_num}: {thread_info['root_id']} ({target_agents} agents)")
        
        # Create directory
        thread_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample agents for this thread based on its actual size
        if len(all_agents) > 0:
            thread_agents = sample_agents_for_thread(
                all_agents, 
                target_agents,
                seed=thread_num
            )
            agents_path = thread_dir / 'agents_for_thread.csv'
            thread_agents.to_csv(agents_path, index=False)
        
        # Create config and metadata
        create_thread_config(thread_dir, thread_info, thread_num)
        create_thread_metadata(thread_dir, thread_info)
    
    # Save manifest
    manifest = {
        'total_threads': actual_count,
        'threads': [
            {
                'thread_num': idx + 1,
                'root_id': str(row['root_id']),
                'internal_replies': int(row['internal_replies']),
                'agents_count': int(row['internal_replies']),
                'max_depth': int(row['max_depth'])
            }
            for idx, row in threads_df.iterrows()
        ]
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"\nPrepared {actual_count} threads in: {output_dir}")
    print(f"Manifest saved: {manifest_path}")


if __name__ == '__main__':
    main()
