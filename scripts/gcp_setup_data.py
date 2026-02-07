"""
GCP Data Setup Script

Downloads august data, calculates thread depths, and extracts only the agents
needed for the top threads. Avoids processing all agents.

Usage:
    python scripts/gcp_setup_data.py [--parts 23 24 25 26 27 28] [--top-threads 50]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np


def download_data_parts(parts: list, data_dir: Path):
    """Clone and download specific parts from the x-24-us-election repo."""
    print("="*60)
    print("STEP 1: Downloading August Data")
    print("="*60)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if git-lfs is installed
    result = subprocess.run(['which', 'git-lfs'], capture_output=True)
    if result.returncode != 0:
        print("Installing git-lfs...")
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'git-lfs'], check=True)
    
    subprocess.run(['git', 'lfs', 'install'], check=True)
    
    # Create sparse checkout folders string
    part_folders = ' '.join([f'part_{p}' for p in parts])
    
    print(f"Cloning parts: {parts}")
    
    # Clone with sparse checkout
    if Path('temp_election_data').exists():
        subprocess.run(['rm', '-rf', 'temp_election_data'])
    
    subprocess.run([
        'git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
        'https://github.com/sinking8/x-24-us-election.git', 'temp_election_data'
    ], check=True)
    
    os.chdir('temp_election_data')
    subprocess.run(['git', 'sparse-checkout', 'set'] + [f'part_{p}' for p in parts], check=True)
    subprocess.run(['git', 'lfs', 'pull'], check=True)
    os.chdir('..')
    
    # Copy CSV files
    for part in parts:
        src = Path(f'temp_election_data/part_{part}')
        for csv_file in src.glob('*.csv'):
            dest = data_dir / csv_file.name
            print(f"  Copying {csv_file.name}...")
            subprocess.run(['cp', str(csv_file), str(dest)], check=True)
    
    # Cleanup
    subprocess.run(['rm', '-rf', 'temp_election_data'])
    
    print(f"\n✓ Downloaded {len(parts)} parts to {data_dir}")


def calculate_thread_depths(data_dir: Path, output_path: Path, top_n: int = 50):
    """Calculate thread depths from august data."""
    print("\n" + "="*60)
    print("STEP 2: Calculating Thread Depths")
    print("="*60)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all august chunks
    all_tweets = []
    for csv_file in sorted(data_dir.glob('aug_chunk_*.csv')):
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file, dtype={'id': str, 'conversationId': str, 'inReplyToTweetId': str, 'userId': str})
        all_tweets.append(df)
    
    if not all_tweets:
        print("ERROR: No august data files found!")
        sys.exit(1)
    
    tweets_df = pd.concat(all_tweets, ignore_index=True)
    print(f"  Loaded {len(tweets_df)} tweets")
    
    # Find conversation sizes
    conv_sizes = tweets_df.groupby('conversationId').size().reset_index(name='total_tweets')
    conv_sizes = conv_sizes[conv_sizes['total_tweets'] >= 10]  # At least 10 tweets
    
    # Calculate internal replies and depths for top conversations
    results = []
    
    for _, row in conv_sizes.nlargest(top_n * 2, 'total_tweets').iterrows():
        conv_id = row['conversationId']
        conv_tweets = tweets_df[tweets_df['conversationId'] == conv_id]
        
        # Internal replies = replies within this conversation
        internal_replies = conv_tweets[conv_tweets['inReplyToTweetId'].isin(conv_tweets['id'])]
        
        # Find root tweet
        root = conv_tweets[conv_tweets['id'] == conv_id]
        if len(root) == 0:
            continue
        
        # Calculate depth distribution
        depth_map = {conv_id: 0}
        for _, tweet in conv_tweets.iterrows():
            parent = tweet.get('inReplyToTweetId', '')
            if pd.notna(parent) and parent in depth_map:
                depth_map[tweet['id']] = depth_map[parent] + 1
        
        depths = list(depth_map.values())
        max_depth = max(depths) if depths else 0
        
        # Count depths
        depth_dist = {}
        for d in depths:
            depth_dist[d] = depth_dist.get(d, 0) + 1
        
        results.append({
            'root_id': conv_id,
            'internal_replies': len(internal_replies),
            'total_tweets': len(conv_tweets),
            'max_depth': max_depth,
            'depth_dist': str(depth_dist),
            'text': root.iloc[0].get('renderedContent', '')[:500] if len(root) > 0 else ''
        })
    
    # Sort by internal replies and take top N
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('internal_replies', ascending=False).head(top_n)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved top {len(results_df)} threads to {output_path}")
    return results_df


def extract_thread_agents(data_dir: Path, threads_df: pd.DataFrame, output_dir: Path):
    """Extract agents only from the specified threads."""
    print("\n" + "="*60)
    print("STEP 3: Extracting Thread Agents")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all conversation IDs from top threads
    thread_conv_ids = set(threads_df['root_id'].astype(str))
    print(f"  Looking for agents in {len(thread_conv_ids)} threads")
    
    # Load all august data and find users in these threads
    all_thread_users = set()
    all_thread_tweets = []
    
    for csv_file in sorted(data_dir.glob('aug_chunk_*.csv')):
        print(f"  Scanning {csv_file.name}...")
        df = pd.read_csv(csv_file, dtype={'id': str, 'conversationId': str, 'userId': str})
        
        # Filter to our threads
        thread_tweets = df[df['conversationId'].isin(thread_conv_ids)]
        
        if len(thread_tweets) > 0:
            all_thread_users.update(thread_tweets['userId'].dropna().unique())
            all_thread_tweets.append(thread_tweets)
    
    print(f"\n  Found {len(all_thread_users)} unique users in target threads")
    
    if not all_thread_tweets:
        print("ERROR: No tweets found for target threads!")
        sys.exit(1)
    
    # Combine thread tweets for DNA extraction
    combined_tweets = pd.concat(all_thread_tweets, ignore_index=True)
    
    # Save intermediate file for DNA extraction
    thread_users_path = output_dir / 'thread_users.csv'
    combined_tweets.to_csv(thread_users_path, index=False)
    print(f"  Saved {len(combined_tweets)} tweets for DNA extraction")
    
    return combined_tweets, list(all_thread_users)


def process_agent_dna(tweets_df: pd.DataFrame, user_ids: list, output_dir: Path):
    """Process DNA for agents using existing extraction logic."""
    print("\n" + "="*60)
    print("STEP 4: Processing Agent DNA")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group tweets by user
    user_tweets = tweets_df.groupby('userId')
    
    agents = []
    for user_id in user_ids:
        if user_id not in user_tweets.groups:
            continue
        
        user_data = user_tweets.get_group(user_id)
        
        # Basic DNA extraction
        tweet_count = len(user_data)
        
        # Aggression proxy: exclamation marks, caps, etc.
        texts = user_data['renderedContent'].fillna('').astype(str)
        avg_caps = texts.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)).mean()
        avg_excl = texts.apply(lambda x: x.count('!') / max(len(x), 1)).mean()
        aggression = min(1.0, (avg_caps * 2 + avg_excl * 10))
        
        # Political label placeholder (would need classifier)
        political_label = 'Center'  # Default, actual classification would happen here
        
        agents.append({
            'user_id': str(user_id),
            'tweet_count': tweet_count,
            'aggression': round(aggression, 3),
            'political_label': political_label,
            'influence': min(1.0, tweet_count / 100),
            'openness': 0.5  # Default
        })
    
    # Save processed agents
    agents_df = pd.DataFrame(agents)
    output_path = output_dir / 'processed_agents_august.csv'
    agents_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Processed {len(agents_df)} agents to {output_path}")
    return agents_df


def main():
    parser = argparse.ArgumentParser(description='Setup data for GCP batch simulations')
    parser.add_argument('--parts', nargs='+', type=int, default=[23, 24, 25, 26, 27, 28],
                       help='Part numbers to download (default: 23-28)')
    parser.add_argument('--top-threads', type=int, default=50,
                       help='Number of top threads to process (default: 50)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download (use existing files)')
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    output_dir = project_root / 'output'
    agents_dir = project_root / 'processed_agents'
    
    print("="*60)
    print("GCP DATA SETUP")
    print("="*60)
    print(f"Parts to download: {args.parts}")
    print(f"Top threads to process: {args.top_threads}")
    
    # Step 1: Download data
    if not args.skip_download:
        download_data_parts(args.parts, data_dir)
    else:
        print("\nSkipping download (using existing data)")
    
    # Step 2: Calculate thread depths
    threads_df = calculate_thread_depths(
        data_dir, 
        output_dir / 'august_thread_depths.csv',
        args.top_threads
    )
    
    # Step 3: Extract agents from threads
    tweets_df, user_ids = extract_thread_agents(data_dir, threads_df, agents_dir)
    
    # Step 4: Process agent DNA
    process_agent_dna(tweets_df, user_ids, agents_dir)
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {output_dir / 'august_thread_depths.csv'}")
    print(f"  - {agents_dir / 'processed_agents_august.csv'}")
    print(f"\nNext: Run simulations with:")
    print(f"  python scripts/prepare_batch_simulations.py")
    print(f"  python scripts/gcp_batch_runner.py --mode parallel --workers 4")


if __name__ == '__main__':
    main()
