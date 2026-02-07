"""
Prepare 100 Medium Threads for Batch Simulation

Finds threads with 30-100 replies from raw data, extracts the ACTUAL
participants, classifies them through 5 RoBERTa models to build agent DNA,
and creates per-thread simulation directories.

Each directory contains:
- config.yaml: Thread-specific simulation config
- thread_metadata.json: Root tweet + temporal events from real data
- agents_for_thread.csv: Classified DNA for actual thread participants

Usage:
    python scripts/prepare_100_threads.py [--limit 100] [--data-dir data]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── Model loading ────────────────────────────────────────────────────────────

MODELS = {
    'political': {
        'model': 'matous-volf/political-leaning-politics',
        'tokenizer': 'launch/POLITICS',
        'labels': {0: 'Left', 1: 'Center', 2: 'Right'},
    },
    'emotion': {
        'model': 'cardiffnlp/twitter-roberta-base-emotion',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-emotion',
        'labels': {0: 'anger', 1: 'joy', 2: 'optimism', 3: 'sadness'},
    },
    'sentiment': {
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'labels': {0: 'negative', 1: 'neutral', 2: 'positive'},
    },
    'hate': {
        'model': 'cardiffnlp/twitter-roberta-base-hate-latest',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-hate-latest',
        'labels': {0: 'not-hate', 1: 'hate'},
    },
    'offensive': {
        'model': 'cardiffnlp/twitter-roberta-base-offensive',
        'tokenizer': 'cardiffnlp/twitter-roberta-base-offensive',
        'labels': {0: 'not-offensive', 1: 'offensive'},
    },
}


def load_models(device: str) -> dict:
    """Load all 5 RoBERTa classification models."""
    loaded = {}
    for name, info in MODELS.items():
        print(f"  Loading {name} model...")
        tokenizer = AutoTokenizer.from_pretrained(info['tokenizer'])
        model = AutoModelForSequenceClassification.from_pretrained(info['model'])
        model.to(device).eval()
        loaded[name] = {'tokenizer': tokenizer, 'model': model, 'labels': info['labels']}
    return loaded


def classify_text(text: str, model_info: dict, device: str) -> tuple:
    """Classify a single text, return (label, score)."""
    tokenizer = model_info['tokenizer']
    model = model_info['model']
    labels = model_info['labels']

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return labels[top_idx], float(probs[top_idx])


def classify_user_tweets(tweets: list[str], models: dict, device: str) -> dict:
    """
    Classify all tweets for a user, aggregate to user-level DNA.
    Returns dict with all DNA columns.
    """
    results = {k: [] for k in ['political', 'emotion', 'sentiment', 'hate', 'offensive']}

    for text in tweets:
        if not text or len(str(text).strip()) < 5:
            continue
        text = str(text)[:512]
        for model_name, model_info in models.items():
            try:
                label, score = classify_text(text, model_info, device)
                results[model_name].append((label, score))
            except Exception:
                pass

    # Aggregate: mode for labels, mean for scores
    dna = {}

    # Political
    if results['political']:
        labels = [r[0] for r in results['political']]
        scores = [r[1] for r in results['political']]
        dna['political_label'] = max(set(labels), key=labels.count)
        dna['political_score'] = float(np.mean(scores))
    else:
        dna['political_label'] = 'Right'
        dna['political_score'] = 0.5

    # Emotion
    if results['emotion']:
        labels = [r[0] for r in results['emotion']]
        scores = [r[1] for r in results['emotion']]
        dna['emotion_label'] = max(set(labels), key=labels.count)
        dna['emotion_score'] = float(np.mean(scores))
    else:
        dna['emotion_label'] = 'anger'
        dna['emotion_score'] = 0.5

    # Sentiment
    if results['sentiment']:
        labels = [r[0] for r in results['sentiment']]
        scores = [r[1] for r in results['sentiment']]
        dna['sentiment_label'] = max(set(labels), key=labels.count)
        dna['sentiment_score'] = float(np.mean(scores))
    else:
        dna['sentiment_label'] = 'neutral'
        dna['sentiment_score'] = 0.5

    # Hate (normalize: 1 = hateful)
    if results['hate']:
        scores = [r[1] if r[0] == 'hate' else 1 - r[1] for r in results['hate']]
        dna['hate_score'] = float(np.mean(scores))
    else:
        dna['hate_score'] = 0.05

    # Offensive (normalize: 1 = offensive)
    if results['offensive']:
        scores = [r[1] if r[0] == 'offensive' else 1 - r[1] for r in results['offensive']]
        dna['offensive_score'] = float(np.mean(scores))
    else:
        dna['offensive_score'] = 0.1

    return dna


# ── Data extraction ──────────────────────────────────────────────────────────

def extract_user_id(user_str: str) -> int | None:
    """Extract numeric user_id from raw user column string."""
    if pd.isna(user_str):
        return None
    m = re.search(r"'id':\s*(\d+)", str(user_str))
    return int(m.group(1)) if m else None


def extract_username(user_str: str) -> str | None:
    """Extract username from raw user column string."""
    if pd.isna(user_str):
        return None
    m = re.search(r"'username':\s*'([^']+)'", str(user_str))
    return m.group(1) if m else None


def find_medium_threads(raw_df: pd.DataFrame, min_replies: int = 30,
                        max_replies: int = 100, limit: int = 100) -> pd.DataFrame:
    """Find threads with min_replies to max_replies tweets, return top N by count."""
    conv_counts = raw_df.groupby('conversationId').size().reset_index(name='tweet_count')
    medium = conv_counts[
        (conv_counts['tweet_count'] >= min_replies) &
        (conv_counts['tweet_count'] <= max_replies)
    ].sort_values('tweet_count', ascending=False).head(limit)
    return medium


def extract_thread_data(raw_df: pd.DataFrame, conv_id: float) -> dict:
    """
    Extract full thread data: root tweet, all replies, user texts.
    Returns dict with root_tweet info, temporal_events, and user_tweets mapping.
    """
    thread = raw_df[raw_df['conversationId'] == conv_id].copy()
    thread['user_id'] = thread['user'].apply(extract_user_id)
    thread['username'] = thread['user'].apply(extract_username)

    # Sort by epoch/date
    if 'epoch' in thread.columns:
        thread = thread.sort_values('epoch')

    # Find root tweet (where id == conversationId, or first tweet)
    root_mask = thread['id'] == conv_id
    if root_mask.any():
        root_row = thread[root_mask].iloc[0]
    else:
        # Sometimes conversationId stored as float - try matching
        root_mask = (thread['id'] - conv_id).abs() < 1
        if root_mask.any():
            root_row = thread[root_mask].iloc[0]
        else:
            root_row = thread.iloc[0]

    root_uid = root_row['user_id']
    root_username = root_row['username'] if pd.notna(root_row.get('username')) else str(root_uid)
    root_text = str(root_row.get('rawContent', root_row.get('text', '')))

    # Build temporal events
    temporal_events = []
    for i, (_, row) in enumerate(thread.iterrows()):
        uid = row['user_id']
        uname = row['username'] if pd.notna(row.get('username')) else str(uid)
        text = str(row.get('rawContent', row.get('text', '')))
        is_root = (i == 0) or (row['id'] == conv_id) or ((row['id'] - conv_id) < 1 if pd.notna(conv_id) else False)

        temporal_events.append({
            'tweet_id': str(row['id']),
            'user_id': uname,
            'text': text,
            'is_root': bool(is_root),
        })

    # Group tweets by user_id for classification
    user_tweets = {}
    for _, row in thread.iterrows():
        uid = row['user_id']
        if pd.isna(uid):
            continue
        uid = int(uid)
        text = str(row.get('rawContent', row.get('text', '')))
        if uid not in user_tweets:
            user_tweets[uid] = {'texts': [], 'username': row.get('username', str(uid)),
                                'tweet_count': 0, 'view_count': 0, 'reply_count': 0}
        user_tweets[uid]['texts'].append(text)
        user_tweets[uid]['tweet_count'] += 1
        # Try to extract view_count
        vc = row.get('viewCount', 0)
        if pd.notna(vc):
            vc_str = str(vc)
            vc_match = re.search(r"'count':\s*'?(\d+)", vc_str)
            if vc_match:
                user_tweets[uid]['view_count'] += int(vc_match.group(1))
        rc = row.get('replyCount', 0)
        if pd.notna(rc):
            try:
                user_tweets[uid]['reply_count'] += int(rc)
            except (ValueError, TypeError):
                pass

    return {
        'conv_id': conv_id,
        'root_text': root_text,
        'root_username': root_username,
        'root_uid': root_uid,
        'temporal_events': temporal_events,
        'user_tweets': user_tweets,
        'total_tweets': len(thread),
        'unique_users': len(user_tweets),
    }


# ── Directory creation ───────────────────────────────────────────────────────

def create_thread_dir(
    thread_dir: Path,
    thread_data: dict,
    agents_df: pd.DataFrame,
    thread_num: int,
    max_rounds: int = 10,
) -> None:
    """Create a complete thread simulation directory."""
    thread_dir.mkdir(parents=True, exist_ok=True)

    # 1. agents_for_thread.csv
    agents_path = thread_dir / 'agents_for_thread.csv'
    agents_df.to_csv(agents_path, index=False)

    # 2. thread_metadata.json
    metadata = {
        'root_tweet': {
            'id': str(thread_data['conv_id']),
            'user_id': thread_data['root_username'],
            'text': thread_data['root_text'],
            'expected_replies': thread_data['total_tweets'] - 1,
        },
        'temporal_events': thread_data['temporal_events'],
        'thread_structure': {
            'max_depth': 3,
        },
    }
    with open(thread_dir / 'thread_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 3. config.yaml
    config = {
        'llm': {
            'provider': 'ollama',
            'model': 'dolphin-llama3:8b',
            'temperature': 0.9,
            'max_tokens': 150,
        },
        'paths': {
            'thread_metadata': str(thread_dir / 'thread_metadata.json'),
            'agents_for_thread': str(thread_dir / 'agents_for_thread.csv'),
        },
        'simulation': {
            'max_rounds': max_rounds,
            'thread_num': thread_num,
        },
        'target_tweet_id': str(thread_data['conv_id']),
        'thread_info': {
            'root_tweet_id': str(thread_data['conv_id']),
            'internal_replies': thread_data['total_tweets'] - 1,
            'max_depth': 3,
            'root_text': thread_data['root_text'][:500],
        },
    }
    with open(thread_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prepare 100 medium threads with classified agents')
    parser.add_argument('--limit', type=int, default=100, help='Number of threads (default: 100)')
    parser.add_argument('--min-replies', type=int, default=30, help='Min replies per thread')
    parser.add_argument('--max-replies', type=int, default=100, help='Max replies per thread')
    parser.add_argument('--max-rounds', type=int, default=10, help='Simulation rounds per thread')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with raw CSV chunks')
    parser.add_argument('--output', type=str, default='batch_simulations_100', help='Output directory')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Step 1: Find threads ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Finding medium threads")
    print("=" * 80)

    # Load raw data from all aug chunks
    all_raw = []
    for chunk_file in sorted(data_dir.glob('aug_chunk_*.csv')):
        print(f"  Loading {chunk_file.name}...")
        df = pd.read_csv(chunk_file)
        all_raw.append(df)

    if not all_raw:
        print("ERROR: No aug_chunk_*.csv files found in", data_dir)
        sys.exit(1)

    raw_df = pd.concat(all_raw, ignore_index=True)
    print(f"  Total tweets: {len(raw_df)}")

    medium_threads = find_medium_threads(raw_df, args.min_replies, args.max_replies, args.limit)
    print(f"  Found {len(medium_threads)} threads with {args.min_replies}-{args.max_replies} replies")

    if len(medium_threads) == 0:
        print("ERROR: No threads found in range")
        sys.exit(1)

    # ── Step 2: Load classification models ───────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: Loading 5 RoBERTa classification models")
    print("=" * 80)

    models = load_models(device)
    print("  All models loaded.")

    # ── Step 3: Process each thread ──────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"STEP 3: Extracting & classifying agents for {len(medium_threads)} threads")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for idx, (_, row) in enumerate(medium_threads.iterrows()):
        conv_id = row['conversationId']
        tweet_count = int(row['tweet_count'])
        thread_num = idx + 1

        print(f"\n{'─' * 60}")
        print(f"Thread {thread_num}/{len(medium_threads)}: {conv_id} ({tweet_count} tweets)")

        # Extract thread data
        thread_data = extract_thread_data(raw_df, conv_id)
        print(f"  Users: {thread_data['unique_users']}, Tweets: {thread_data['total_tweets']}")

        # Classify each user
        agent_rows = []
        user_items = list(thread_data['user_tweets'].items())
        for uid, uinfo in tqdm(user_items, desc="  Classifying", leave=False):
            dna = classify_user_tweets(uinfo['texts'], models, device)
            dna['user_id'] = uid
            dna['tweet_count'] = uinfo['tweet_count']
            dna['view_count'] = uinfo['view_count']
            dna['reply_count'] = uinfo['reply_count']
            agent_rows.append(dna)

        agents_df = pd.DataFrame(agent_rows)
        # Reorder columns
        col_order = ['user_id', 'tweet_count', 'view_count', 'reply_count',
                      'political_label', 'political_score', 'emotion_label', 'emotion_score',
                      'sentiment_label', 'sentiment_score', 'hate_score', 'offensive_score']
        agents_df = agents_df[[c for c in col_order if c in agents_df.columns]]

        # Compute aggression for summary
        agents_df['aggression'] = agents_df['hate_score'] + agents_df['offensive_score']

        # Summary
        pol_dist = agents_df['political_label'].value_counts(normalize=True)
        print(f"  Political: {dict(pol_dist.round(3))}")
        print(f"  Mean aggression: {agents_df['aggression'].mean():.3f}")

        # Create thread directory
        thread_dir = output_dir / f'thread_{thread_num:03d}'
        # Drop aggression column before saving (it's computed from hate+offensive)
        save_df = agents_df.drop(columns=['aggression'], errors='ignore')
        create_thread_dir(thread_dir, thread_data, save_df, thread_num, args.max_rounds)
        print(f"  Saved: {thread_dir}")

        manifest_entries.append({
            'thread_num': int(thread_num),
            'conv_id': str(conv_id),
            'tweet_count': int(tweet_count),
            'unique_users': int(thread_data['unique_users']),
            'agents_classified': int(len(agents_df)),
            'political_dist': {k: int(v) for k, v in agents_df['political_label'].value_counts().items()},
            'mean_aggression': round(float(agents_df['aggression'].mean()), 3),
        })

    # Save manifest
    manifest = {
        'total_threads': len(manifest_entries),
        'reply_range': f'{args.min_replies}-{args.max_replies}',
        'threads': manifest_entries,
    }
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PREPARATION COMPLETE: {len(manifest_entries)} threads in {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
