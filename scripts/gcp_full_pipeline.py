#!/usr/bin/env python3
"""
GCP Full Pipeline - Complete Batch Simulation System

This single script handles the entire pipeline:
1. Downloads august data (parts 23-28, 117 chunks)
2. Finds top 100 threads by internal reply count
3. Extracts agents who participated in those threads
4. Classifies agents through 5 models (political, emotion, sentiment, hate, offensive)
5. Prepares simulation directories
6. Runs simulations with validation
7. Aggregates results

Usage:
    python scripts/gcp_full_pipeline.py [--step STEP] [--threads N] [--workers N]

Steps:
    1 = Download data only
    2 = Find threads only
    3 = Extract and classify agents
    4 = Prepare simulations
    5 = Run simulations
    6 = Aggregate results
    all = Run entire pipeline (default)

Example:
    python scripts/gcp_full_pipeline.py --step all --threads 100 --workers 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
AGENTS_DIR = PROJECT_ROOT / 'processed_agents'
BATCH_SIM_DIR = PROJECT_ROOT / 'batch_simulations'
BATCH_OUTPUT_DIR = PROJECT_ROOT / 'batch_output'

GITHUB_REPO = 'https://github.com/sinking8/x-24-us-election.git'
PARTS_TO_DOWNLOAD = [23, 24, 25, 26, 27, 28]  # Contains chunks 1-117


# =============================================================================
# STEP 1: DOWNLOAD AUGUST DATA
# =============================================================================
def download_august_data():
    """Download parts 23-28 from the election repo."""
    print("\n" + "="*80)
    print("STEP 1: DOWNLOADING AUGUST DATA")
    print("="*80)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data
    existing = list(DATA_DIR.glob('aug_chunk_*.csv'))
    if len(existing) >= 100:
        print(f"Found {len(existing)} existing chunks. Skipping download.")
        return True
    
    # Install git-lfs if needed
    result = subprocess.run(['which', 'git-lfs'], capture_output=True)
    if result.returncode != 0:
        print("Installing git-lfs...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'git-lfs'], check=True)
    
    subprocess.run(['git', 'lfs', 'install'], check=True)
    
    # Clone with sparse checkout
    temp_dir = PROJECT_ROOT / 'temp_election_data'
    if temp_dir.exists():
        subprocess.run(['rm', '-rf', str(temp_dir)], check=True)
    
    part_folders = [f'part_{p}' for p in PARTS_TO_DOWNLOAD]
    print(f"Cloning parts: {PARTS_TO_DOWNLOAD}")
    
    subprocess.run([
        'git', 'clone', '--filter=blob:none', '--sparse',
        GITHUB_REPO, str(temp_dir)
    ], check=True)
    
    subprocess.run(['git', 'sparse-checkout', 'set'] + part_folders, check=True, cwd=str(temp_dir))
    subprocess.run(['git', 'lfs', 'pull'], check=True, cwd=str(temp_dir))
    
    # Decompress and copy files
    print("Decompressing and copying files...")
    for part in PARTS_TO_DOWNLOAD:
        part_dir = temp_dir / f'part_{part}'
        if part_dir.exists():
            # Decompress .gz files
            gz_files = list(part_dir.glob('*.gz'))
            if gz_files:
                subprocess.run(['gunzip'] + [str(f) for f in gz_files], check=True)
            
            # Copy CSV files
            for csv_file in part_dir.glob('aug_chunk_*.csv'):
                dest = DATA_DIR / csv_file.name
                subprocess.run(['cp', str(csv_file), str(dest)], check=True)
                print(f"  Copied: {csv_file.name}")
    
    # Cleanup
    subprocess.run(['rm', '-rf', str(temp_dir)], check=True)
    
    # Verify
    chunks = list(DATA_DIR.glob('aug_chunk_*.csv'))
    print(f"\n✓ Downloaded {len(chunks)} august chunks")
    return len(chunks) > 0


# =============================================================================
# STEP 2: FIND TOP THREADS
# =============================================================================
def find_top_threads(top_n: int = 100):
    """
    Find top N threads by internal reply count using a 2-pass memory efficient approach.
    Pass 1: Count total tweets per conversation to find candidates.
    Pass 2: Load only data for candidate conversations to analyze structure.
    """
    print("\n" + "="*80)
    print("STEP 2: FINDING TOP THREADS (Memory Optimized)")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'august_thread_depths.csv'
    
    chunks = sorted(DATA_DIR.glob('aug_chunk_*.csv'))
    print(f"Processing {len(chunks)} chunks...")

    # PASS 1: Count conversation sizes
    print("\nPass 1: Counting conversation sizes...")
    conv_counts = {}
    
    for chunk_file in tqdm(chunks, desc="Scanning chunks"):
        try:
            # Only read conversationId column for speed/memory
            df = pd.read_csv(chunk_file, usecols=['conversationId'], dtype={'conversationId': str})
            counts = df['conversationId'].value_counts()
            
            for conv_id, count in counts.items():
                conv_counts[conv_id] = conv_counts.get(conv_id, 0) + count
                
        except Exception as e:
            print(f"  Warning: Error reading {chunk_file.name}: {e}")

    # Filter candidates (at least 10 tweets)
    candidates = {k: v for k, v in conv_counts.items() if v >= 10}
    print(f"Found {len(candidates)} conversations with 10+ tweets")
    
    # Select top candidates to analyze (get 3x top_n to be safe, sorted by total tweets)
    # We use total tweets as a proxy for internal replies for the first pass
    candidate_ids = set(sorted(candidates, key=candidates.get, reverse=True)[:top_n * 3])
    print(f"Selected {len(candidate_ids)} candidates for detailed analysis")

    # PASS 2: Load data for candidates only
    print("\nPass 2: Loading candidate thread data...")
    candidate_tweets = []
    
    for chunk_file in tqdm(chunks, desc="Extracting threads"):
        try:
            # Read minimal columns, handle missing ones
            df = pd.read_csv(chunk_file, dtype={
                'id': str, 'conversationId': str,
                'text': str, 'renderedContent': str, 'rawContent': str
            })

            # Extract userId from 'user' dictionary column (August data)
            if 'userId' not in df.columns:
                if 'user' in df.columns:
                    df['userId'] = df['user'].apply(lambda x:
                        re.search(r"'id':\s*(\d+)", str(x)).group(1) if pd.notna(x) and re.search(r"'id':\s*(\d+)", str(x)) else None
                    )
                else:
                    df['userId'] = np.nan

            # Map August column to expected column name
            if 'inReplyToTweetId' not in df.columns:
                if 'in_reply_to_status_id_str' in df.columns:
                    df['inReplyToTweetId'] = df['in_reply_to_status_id_str'].astype(str)
                else:
                    df['inReplyToTweetId'] = np.nan
            else:
                df['inReplyToTweetId'] = df['inReplyToTweetId'].astype(str)
            
            # Filter to just our candidates
            mask = df['conversationId'].isin(candidate_ids)
            if mask.any():
                candidate_tweets.append(df[mask])
                
        except Exception as e:
             pass

    if not candidate_tweets:
        print("Error: No candidate tweets found in pass 2!")
        return pd.DataFrame()

    tweets_df = pd.concat(candidate_tweets, ignore_index=True)
    print(f"Loaded {len(tweets_df)} tweets for detailed analysis")
    
    # Analyze structure (same logic as before, but on smaller dataset)
    results = []
    
    for conv_id in tqdm(candidate_ids, desc="Analyzing structure"):
        conv_tweets = tweets_df[tweets_df['conversationId'] == conv_id]
        
        if len(conv_tweets) < 10: 
            continue

        # Internal replies = replies within this conversation
        internal_ids = set(conv_tweets['id'].dropna().astype(str))
        
        if 'inReplyToTweetId' in conv_tweets.columns:
            # Safe check ensuring matching types
            internal_replies = conv_tweets[conv_tweets['inReplyToTweetId'].isin(internal_ids)]
            internal_replies_count = len(internal_replies)
        else:
            internal_replies_count = 0
        
        # Find root tweet
        root = conv_tweets[conv_tweets['id'] == conv_id]
        if len(root) == 0:
            continue
        
        # Calculate depth
        depth_map = {conv_id: 0}
        # Sort by id/time to ensure parents processed before children usually
        # But for depth map building, iterative approach works if order isn't guaranteed
        # We'll just do a multi-pass or recursive approach implicitly by iterating
        # Actually simplest is to build adjacency and traverse
        
        # Build adjacency for robust depth calc
        children_map = {}
        if 'inReplyToTweetId' in conv_tweets.columns:
            for _, tweet in conv_tweets.iterrows():
                pid = tweet['inReplyToTweetId']
                if pd.notna(pid):
                    children_map.setdefault(str(pid), []).append(str(tweet['id']))

        # BFS for depth
        queue = [(conv_id, 0)]
        depth_map = {conv_id: 0}
        max_depth = 0
        
        processed_count = 0 
        while processed_count < len(queue):
            curr_id, curr_depth = queue[processed_count]
            processed_count += 1
            max_depth = max(max_depth, curr_depth)
            
            if curr_id in children_map:
                for child_id in children_map[curr_id]:
                    if child_id not in depth_map: # Avoid cycles
                        depth_map[child_id] = curr_depth + 1
                        queue.append((child_id, curr_depth + 1))
        
        depths = list(depth_map.values())
        
        # Count depth distribution
        depth_dist = {}
        for d in depths:
            depth_dist[d] = depth_dist.get(d, 0) + 1
        
        results.append({
            'root_id': conv_id,
            'internal_replies': internal_replies_count,
            'total_tweets': len(conv_tweets),
            'max_depth': max_depth,
            'depth_dist': str(depth_dist),
            'text': str(root.iloc[0].get('renderedContent', ''))[:500] if len(root) > 0 else ''
        })
    
    # Sort and save top N
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        error_msg = "ERROR: No threads found with 10+ tweets!"
        print(error_msg)
        raise ValueError(error_msg)

    results_df = results_df.sort_values('internal_replies', ascending=False).head(top_n)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved top {len(results_df)} threads to {output_path}")
    print(f"  Total internal replies: {results_df['internal_replies'].sum()}")
    print(f"  Avg internal replies: {results_df['internal_replies'].mean():.1f}")
    return results_df


# =============================================================================
# STEP 3: EXTRACT AND CLASSIFY AGENTS
# =============================================================================
def extract_and_classify_agents(threads_df: pd.DataFrame):
    """Extract agents from threads and classify through 5 models."""
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING AND CLASSIFYING AGENTS")
    print("="*80)

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGENTS_DIR / 'processed_agents_august.csv'

    # Check for existing - but VALIDATE it has real data
    if output_path.exists():
        df = pd.read_csv(output_path)
        # Validate: must have at least 100 agents with real user IDs
        valid_agents = df[df['user_id'].notna() & (df['user_id'] != 'nan')]
        if len(valid_agents) >= 100:
            print(f"✓ Found existing processed agents ({len(valid_agents)} valid agents). Skipping.")
            return df
        else:
            print(f"⚠ Found existing file but only {len(valid_agents)} valid agents (need 100+)")
            print(f"  Deleting bad file and re-extracting...")
            output_path.unlink()  # Delete the bad file
    
    # Get thread IDs
    thread_ids = set(threads_df['root_id'].astype(str))
    print(f"Finding agents in {len(thread_ids)} threads...")
    
    # Load tweets from target threads
    chunks = sorted(DATA_DIR.glob('aug_chunk_*.csv'))
    thread_tweets = []
    
    for chunk_file in tqdm(chunks, desc="Scanning chunks"):
        try:
            # Read minimal columns, handle missing
            df = pd.read_csv(chunk_file, dtype={
                'id': str, 'conversationId': str,
                'text': str, 'renderedContent': str, 'rawContent': str
            })

            # Extract userId from 'user' dictionary column (August data)
            if 'userId' not in df.columns:
                if 'user' in df.columns:
                    # Parse user dict: "{'id': 1234567890, ...}" -> 1234567890
                    df['userId'] = df['user'].apply(lambda x:
                        re.search(r"'id':\s*(\d+)", str(x)).group(1) if pd.notna(x) and re.search(r"'id':\s*(\d+)", str(x)) else None
                    )
                else:
                    df['userId'] = np.nan

            matched = df[df['conversationId'].isin(thread_ids)]
            if len(matched) > 0:
                thread_tweets.append(matched)
        except Exception as e:
            # print(f"  Warning: {e}")
            pass
    
    if not thread_tweets:
        print("ERROR: No tweets found for target threads!")
        return None
    
    all_thread_tweets = pd.concat(thread_tweets, ignore_index=True)

    # Verify we extracted user IDs
    unique_users = all_thread_tweets['userId'].dropna().unique()
    print(f"Found {len(unique_users)} unique users in target threads")
    print(f"Total tweets to process: {len(all_thread_tweets)}")

    if len(unique_users) == 0:
        print("ERROR: No valid user IDs extracted!")
        print("Sample of 'user' column values:")
        print(all_thread_tweets['user'].head(3) if 'user' in all_thread_tweets.columns else "No 'user' column found")
        return None
    
    # Load classification models
    print("\nLoading 5 classification models...")
    import torch
    from transformers import pipeline
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using: {'GPU' if device == 0 else 'CPU'}")
    
    political_p = pipeline("text-classification", model="matous-volf/political-leaning-politics", 
                          tokenizer="launch/POLITICS", device=device, truncation=True, max_length=512)
    emo_p = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", 
                     device=device, truncation=True, max_length=512)
    sent_p = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                      device=device, truncation=True, max_length=512)
    hate_p = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest", 
                      device=device, truncation=True, max_length=512)
    offen_p = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", 
                       device=device, truncation=True, max_length=512)
    print("✓ Models loaded")
    
    POLITICAL_LABEL_MAP = {'LABEL_0': 'Left', 'LABEL_1': 'Center', 'LABEL_2': 'Right'}
    
    def get_agent_dna(text):
        """Extract DNA from text using 5 models."""
        text = str(text)[:512]
        try:
            p = political_p(text)[0]
            e = emo_p(text)[0]
            sn = sent_p(text)[0]
            h = hate_p(text)[0]
            o = offen_p(text)[0]
            
            return {
                'political_label': POLITICAL_LABEL_MAP.get(p['label'], p['label']),
                'political_score': p['score'],
                'emotion_label': e['label'],
                'emotion_score': e['score'],
                'sentiment_label': sn['label'],
                'sentiment_score': sn['score'],
                'hate_score': h['score'] if h['label'] == 'HATE' else 1 - h['score'],
                'offensive_score': o['score'] if o['label'] == 'OFFENSIVE' else 1 - o['score']
            }
        except Exception as e:
            return {
                'political_label': 'Center', 'political_score': 0.5,
                'emotion_label': 'neutral', 'emotion_score': 0.5,
                'sentiment_label': 'neutral', 'sentiment_score': 0.5,
                'hate_score': 0.0, 'offensive_score': 0.0
            }
    
    
    # Checkpoint setup
    CHECKPOINT_FILE = AGENTS_DIR / 'agent_classification_chk_v3.csv'
    processed_ids = set()
    
    if CHECKPOINT_FILE.exists():
        try:
            print(f"\nChecking for existing checkpoints in {CHECKPOINT_FILE.name}...")
            # Load processed IDs to skip
            chk_df = pd.read_csv(CHECKPOINT_FILE, usecols=['tweet_id'], dtype={'tweet_id': str})
            processed_ids = set(chk_df['tweet_id'])
            print(f"✓ Found {len(processed_ids)} already processed tweets. Resuming...")
        except Exception as e:
            print(f"⚠ Warning: Could not read checkpoint file: {e}")

    # Classify each tweet
    print("\nClassifying tweets...")
    
    # Buffer for batch saving
    batch_results = []
    batch_size = 100
    
    for idx, row in tqdm(all_thread_tweets.iterrows(), total=len(all_thread_tweets), desc="Classifying"):
        tweet_id = str(row.get('id', ''))
        
        # Skip if already done
        if tweet_id in processed_ids:
            continue
            
        # Try rawContent (August), then renderedContent (May-July), then text (fallback)
        # Use pd.notna to avoid NaN becoming 'nan' string
        _raw = row.get('rawContent')
        _rendered = row.get('renderedContent')
        _fallback = row.get('text', '')
        text = _raw if pd.notna(_raw) else (_rendered if pd.notna(_rendered) else (_fallback if pd.notna(_fallback) else ''))
        dna = get_agent_dna(text)
        
        # Extract view count
        try:
            match = re.search(r"'count':\s*'(\d+)'", str(row.get('viewCount', '')))
            view_count = int(match.group(1)) if match else 0
        except:
            view_count = 0
        
        result = {
            'user_id': str(row.get('userId', 'unknown')),
            'tweet_id': tweet_id,
            'conversation_id': str(row.get('conversationId', '')),
            'view_count': view_count,
            'reply_count': int(row.get('replyCount', 0)) if pd.notna(row.get('replyCount')) else 0,
            **dna
        }
        
        batch_results.append(result)
        
        # Save batch
        if len(batch_results) >= batch_size:
            df_batch = pd.DataFrame(batch_results)
            write_header = not CHECKPOINT_FILE.exists()
            df_batch.to_csv(CHECKPOINT_FILE, mode='a', header=write_header, index=False)
            batch_results = []
            
    # Save remaining
    if batch_results:
        df_batch = pd.DataFrame(batch_results)
        write_header = not CHECKPOINT_FILE.exists()
        df_batch.to_csv(CHECKPOINT_FILE, mode='a', header=write_header, index=False)

    
    # Aggregate by user
    # Aggregate by user
    print("\nAggregating to user profiles...")
    
    # Load all results from checkpoint
    if not CHECKPOINT_FILE.exists():
        print("ERROR: No checkpoint file found after processing!")
        return None
        
    tweets_df = pd.read_csv(CHECKPOINT_FILE)
    
    def most_common(series):
        mode = series.mode()
        return mode[0] if len(mode) > 0 else series.iloc[0]
    
    user_profiles = tweets_df.groupby('user_id').agg({
        'tweet_id': 'count',
        'view_count': 'sum',
        'reply_count': 'sum',
        'political_score': 'mean',
        'emotion_score': 'mean',
        'sentiment_score': 'mean',
        'hate_score': 'mean',
        'offensive_score': 'mean',
        'political_label': most_common,
        'emotion_label': most_common,
        'sentiment_label': most_common
    }).reset_index()
    
    user_profiles.rename(columns={'tweet_id': 'tweet_count'}, inplace=True)
    
    # Calculate aggression from hate + offensive scores
    user_profiles['aggression'] = (user_profiles['hate_score'] + user_profiles['offensive_score']) / 2

    # Validate we got real data
    valid_agents = user_profiles[user_profiles['user_id'].notna() & (user_profiles['user_id'] != 'nan')]
    if len(valid_agents) < 100:
        error_msg = f"ERROR: Only extracted {len(valid_agents)} valid agents (expected 100+)"
        print(error_msg)
        raise ValueError(error_msg)

    # Save
    user_profiles.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(user_profiles)} agent profiles to {output_path}")
    print(f"  Valid agents: {len(valid_agents)}")
    print(f"  Political: {user_profiles['political_label'].value_counts().to_dict()}")

    return user_profiles


# =============================================================================
# STEP 4: PREPARE SIMULATIONS
# =============================================================================
def extract_thread_tweets(thread_id: str) -> list:
    """Extract all tweets for a specific thread from August data."""
    chunks = sorted(DATA_DIR.glob('aug_chunk_*.csv'))
    thread_tweets = []

    for chunk_file in chunks:
        try:
            df = pd.read_csv(chunk_file, dtype={
                'id': str, 'conversationId': str,
                'text': str, 'rawContent': str, 'renderedContent': str
            })

            matched = df[df['conversationId'] == thread_id]
            if len(matched) > 0:
                for _, row in matched.iterrows():
                    # Extract userId from user dict if needed
                    user_id = None
                    if 'user' in row and pd.notna(row['user']):
                        match = re.search(r"'id':\s*(\d+)", str(row['user']))
                        user_id = match.group(1) if match else None

                    _raw = row.get('rawContent')
                    _rendered = row.get('renderedContent')
                    _fallback = row.get('text', '')
                    safe_text = _raw if pd.notna(_raw) else (_rendered if pd.notna(_rendered) else (_fallback if pd.notna(_fallback) else ''))

                    tweet_data = {
                        'tweet_id': str(row['id']),
                        'user_id': user_id,
                        'text': str(safe_text),
                        'is_root': str(row['id']) == thread_id
                    }
                    thread_tweets.append(tweet_data)
        except Exception as e:
            pass  # Skip problematic chunks

    return thread_tweets


def prepare_simulations(threads_df: pd.DataFrame, agents_df: pd.DataFrame):
    """Prepare simulation directories for each thread."""
    print("\n" + "="*80)
    print("STEP 4: PREPARING SIMULATIONS")
    print("="*80)

    # Validate inputs
    if agents_df is None or len(agents_df) == 0:
        error_msg = "ERROR: No agents provided to prepare_simulations!"
        print(error_msg)
        raise ValueError(error_msg)

    valid_agents = agents_df[agents_df['user_id'].notna() & (agents_df['user_id'] != 'nan')]
    if len(valid_agents) < 100:
        error_msg = f"ERROR: Only {len(valid_agents)} valid agents (need 100+). Cannot prepare simulations."
        print(error_msg)
        raise ValueError(error_msg)

    print(f"Using {len(valid_agents)} valid agents for simulations")

    print(f"Using {len(valid_agents)} valid agents for simulations")

    # Select top 100 but sort Smallest -> Largest for faster initial feedback
    # We take top 100 by size, THEN sort them ascending
    threads_df = threads_df.nlargest(100, 'internal_replies').sort_values('internal_replies', ascending=True)

    BATCH_SIM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if we should clean up (only if re-running Step 4 explicitly)
    # But usually safe to just overwrite.
    
    print("\nExtracting thread tweet data (optimized single pass)...")
    
    # Optimization: Pre-load all tweets for target threads in one pass
    target_thread_ids = set(threads_df['root_id'].astype(str))
    thread_tweets_map = {tid: [] for tid in target_thread_ids}
    
    chunks = sorted(DATA_DIR.glob('aug_chunk_*.csv'))
    
    total_found = 0
    pbar = tqdm(chunks, desc="Scanning chunks")
    
    for chunk_file in pbar:
        try:
            df = pd.read_csv(chunk_file, dtype={
                'id': str, 'conversationId': str,
                'text': str, 'rawContent': str, 'renderedContent': str
            })
            
            # Filter for our threads
            matched = df[df['conversationId'].isin(target_thread_ids)]
            
            for _, row in matched.iterrows():
                cid = str(row['conversationId'])
                
                # Extract userId from user dict if needed
                user_id = None
                if 'user' in row and pd.notna(row['user']):
                    match = re.search(r"'id':\s*(\d+)", str(row['user']))
                    user_id = match.group(1) if match else None

                # Use pd.notna to avoid NaN becoming 'nan' string
                raw = row.get('rawContent')
                rendered = row.get('renderedContent')
                fallback_text = row.get('text', '')
                safe_text = raw if pd.notna(raw) else (rendered if pd.notna(rendered) else (fallback_text if pd.notna(fallback_text) else ''))

                tweet_data = {
                    'tweet_id': str(row['id']),
                    'user_id': user_id,
                    'text': str(safe_text),
                    'is_root': str(row['id']) == cid
                }

                if cid in thread_tweets_map:
                    thread_tweets_map[cid].append(tweet_data)
                    total_found += 1
            
            pbar.set_postfix({'tweets_found': total_found})
                    
        except Exception as e:
            pass # Skip problematic chunks

    print("Generating simulation files...")
    
    # Reset index to ensure clean iteration, though enumerate is safer
    threads_df = threads_df.reset_index(drop=True)
    
    for i, row in threads_df.iterrows():
        thread_num = i + 1
        thread_dir = BATCH_SIM_DIR / f'thread_{thread_num:03d}'
        thread_dir.mkdir(parents=True, exist_ok=True)
        
        target_agents = int(row['internal_replies'])

        # Sample agents
        if len(agents_df) > 0:
            n_agents = min(target_agents, len(agents_df))
            thread_agents = agents_df.sample(n=n_agents, random_state=thread_num)
            thread_agents.to_csv(thread_dir / 'agents_for_thread.csv', index=False)

        # Get actual tweet data from map and resolve root text
        temporal_events = thread_tweets_map.get(str(row['root_id']), [])

        root_user_id = str(row.get('userId', 'unknown'))
        root_text = str(row['text']) if pd.notna(row['text']) else ''
        for t in temporal_events:
            if t.get('is_root'):
                if root_user_id in ('unknown', 'nan'):
                    root_user_id = t.get('user_id', root_user_id)
                # Always prefer actual tweet text over CSV truncated text
                if t.get('text') and t['text'] != 'nan':
                    root_text = t['text']
                break

        # Create config (uses resolved root_text)
        config = {
            'target_tweet_id': str(row['root_id']),
            'paths': {
                'thread_metadata': str(thread_dir / 'thread_metadata.json'),
                'agents_for_thread': str(thread_dir / 'agents_for_thread.csv'),
            },
            'llm': {
                'provider': 'ollama',
                'model': 'dolphin-llama3:8b',
                'temperature': 1.1,
                'max_tokens': 150,
                'system_prompt': "You are a Twitter user engaging in a political discussion. Your persona: Political Leaning: {political_label}, Aggression Level: {aggression}, Emotion: {emotion}. Write a short, realistic tweet reply (under 280 chars). Do not use hashtags unless necessary. Be casual.",
            },
            'thread_info': {
                'root_tweet_id': str(row['root_id']),
                'internal_replies': int(row['internal_replies']),
                'max_depth': int(row['max_depth']),
                'root_text': root_text[:500]
            },
            'simulation': {
                'max_rounds': 10,
                'thread_num': thread_num
            }
        }

        import yaml
        with open(thread_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create metadata with temporal_events for validation
        metadata = {
            'root_tweet': {
                'id': str(row['root_id']),
                'user_id': str(root_user_id),
                'text': root_text,
                'expected_replies': int(row['internal_replies']),
            },
            'thread_structure': {
                'max_depth': int(row['max_depth']),
                'depth_distribution': str(row['depth_dist'])
            },
            'temporal_events': temporal_events  # Add actual tweet data
        }

        with open(thread_dir / 'thread_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Prepared thread {thread_num}: {row['root_id']} ({len(temporal_events)} tweets, {target_agents} agents)")
    
    # Save manifest
    manifest = {
        'total_threads': len(threads_df),
        'threads': [
            {
                'thread_num': idx + 1,
                'root_id': str(row['root_id']),
                'internal_replies': int(row['internal_replies'])
            }
            for idx, row in threads_df.iterrows()
        ]
    }
    
    with open(BATCH_SIM_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Prepared {len(threads_df)} simulation directories")


# =============================================================================
# STEP 5: RUN SIMULATIONS
# =============================================================================
def run_simulations(workers: int = 4):
    """Run all simulations using the batch runner."""
    print("\n" + "="*80)
    print("STEP 5: RUNNING SIMULATIONS")
    print("="*80)
    
    cmd = [
        sys.executable,
        'scripts/gcp_batch_runner.py',
        '--mode', 'sequential',
        '--rounds', '10'
    ]
    
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print("\n✓ Simulations complete")


# =============================================================================
# STEP 6: AGGREGATE RESULTS
# =============================================================================
def aggregate_results():
    """Aggregate all results into final output."""
    print("\n" + "="*80)
    print("STEP 6: AGGREGATING RESULTS")
    print("="*80)
    
    cmd = [sys.executable, 'scripts/aggregate_results.py']
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    
    print("\n✓ Aggregation complete")
    print(f"\nFinal outputs:")
    print(f"  - {BATCH_OUTPUT_DIR / 'aggregated_results.csv'}")
    print(f"  - {BATCH_OUTPUT_DIR / 'aggregated_statistics.json'}")
    print(f"  - {AGENTS_DIR / 'processed_agents_august.csv'}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='GCP Full Simulation Pipeline')
    parser.add_argument('--step', type=str, default='all',
                       help='Step to run: 1-6 or "all"')
    parser.add_argument('--threads', type=int, default=100,
                       help='Number of top threads to process')
    parser.add_argument('--workers', type=int, default=4,
                       help='Parallel workers for simulation')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GCP FULL SIMULATION PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Step: {args.step}")
    print(f"Top threads: {args.threads}")
    print(f"Workers: {args.workers}")
    
    steps_to_run = []
    if args.step == 'all':
        steps_to_run = [1, 2, 3, 4, 5, 6]
    else:
        steps_to_run = [int(args.step)]
    
    threads_df = None
    agents_df = None
    
    try:
        for step in steps_to_run:
            if step == 1:
                success = download_august_data()
                if not success:
                    raise ValueError("Step 1 failed: No data downloaded")
            elif step == 2:
                threads_df = find_top_threads(args.threads)
                if threads_df is None or len(threads_df) == 0:
                    raise ValueError("Step 2 failed: No threads found")
            elif step == 3:
                if threads_df is None:
                    print("Loading threads from previous run...")
                    threads_df = pd.read_csv(OUTPUT_DIR / 'august_thread_depths.csv')
                    if len(threads_df) == 0:
                        raise ValueError("No threads found in august_thread_depths.csv")
                agents_df = extract_and_classify_agents(threads_df)
                if agents_df is None or len(agents_df) == 0:
                    raise ValueError("Step 3 failed: No agents extracted")
            elif step == 4:
                if threads_df is None:
                    print("Loading threads from previous run...")
                    threads_df = pd.read_csv(OUTPUT_DIR / 'august_thread_depths.csv')
                if agents_df is None:
                    print("Loading agents from previous run...")
                    agents_df = pd.read_csv(AGENTS_DIR / 'processed_agents_august.csv')
                    # Validate loaded agents
                    valid = agents_df[agents_df['user_id'].notna() & (agents_df['user_id'] != 'nan')]
                    if len(valid) < 100:
                        raise ValueError(f"Step 4 failed: Only {len(valid)} valid agents in file (need 100+)")
                prepare_simulations(threads_df, agents_df)
            elif step == 5:
                run_simulations(args.workers)
            elif step == 6:
                aggregate_results()
    except Exception as e:
        print(f"\n{'='*80}")
        print("PIPELINE FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
