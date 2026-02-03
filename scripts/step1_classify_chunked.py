"""
Agent classification pipeline with checkpoint support.
Processes a specified chunk file and saves progress incrementally.

Usage:
    python step1_classify_chunked.py <chunk_number>

Example:
    python step1_classify_chunked.py 2  # Process chunk 2
"""
import pandas as pd
import numpy as np
import re
import torch
import sys
import os
from transformers import pipeline
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
if len(sys.argv) < 2:
    print("Usage: python step1_classify_chunked.py <chunk_number>")
    print("Example: python step1_classify_chunked.py 2")
    sys.exit(1)

CHUNK_NUM = sys.argv[1]
INPUT_FILE = f'data/may_july_chunk_{CHUNK_NUM}.csv'
OUTPUT_FILE = f'processed_agents/processed_agents_chunk_{CHUNK_NUM}.csv'
CHECKPOINT_FILE = f'processed_agents/checkpoint_chunk_{CHUNK_NUM}.json'
CHECKPOINT_INTERVAL = 1000  # Save every 1000 tweets

# Ensure output directory exists
os.makedirs('processed_agents', exist_ok=True)

print(f"Processing: {INPUT_FILE}")
print(f"Output will be saved to: {OUTPUT_FILE}")
print(f"Checkpoint file: {CHECKPOINT_FILE}")

# ============================================================================
# 1. SETUP MODELS (Local & Free)
# ============================================================================
device = 0 if torch.cuda.is_available() else -1
print(f"\nUsing: {'GPU' if device == 0 else 'CPU'}")

print("\nLoading models...")
political_p = pipeline("text-classification", model="matous-volf/political-leaning-politics", tokenizer="launch/POLITICS", device=device)
emo_p       = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion", device=device)
sent_p      = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)
hate_p      = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate-latest", device=device)
offen_p     = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive", device=device)
print("✓ Models loaded")

# Label mapping for political model
POLITICAL_LABEL_MAP = {'LABEL_0': 'Left', 'LABEL_1': 'Center', 'LABEL_2': 'Right'}

def get_agent_dna(text):
    """Extract political leaning, emotion, sentiment, hate, and offensive scores from text."""
    text = str(text)[:512]  # Transformers limit

    # Raw Model Calls
    p = political_p(text)[0]
    e = emo_p(text)[0]
    sn = sent_p(text)[0]
    h = hate_p(text)[0]
    o = offen_p(text)[0]

    return {
        'political_label': POLITICAL_LABEL_MAP.get(p['label'], p['label']),  # Left, Center, Right
        'political_score': p['score'],   # Probability
        'emotion_label': e['label'],  # anger, joy, etc.
        'emotion_score': e['score'],
        'sentiment_label': sn['label'], # positive, neutral, negative
        'sentiment_score': sn['score'],
        'hate_score': h['score'] if h['label'] == 'HATE' else 1 - h['score'],
        'offensive_score': o['score'] if o['label'] == 'OFFENSIVE' else 1 - o['score']
    }

# ============================================================================
# 2. LOAD DATA
# ============================================================================
print(f"\nLoading tweets from {INPUT_FILE}...")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded {len(df)} tweets")

# ============================================================================
# 3. EXTRACT USER IDs
# ============================================================================
print("Extracting user IDs...")
def extract_user_id(user_str):
    """Parse user dict and extract user_id using regex (handles datetime objects)."""
    try:
        # Extract 'id': <number> from the string using regex
        # Handles cases where ast.literal_eval fails due to datetime objects
        match = re.search(r"'id':\s*(\d+)", str(user_str))
        if match:
            return int(match.group(1))
        return None
    except:
        return None

df['user_id'] = df['user'].apply(extract_user_id)
df = df.dropna(subset=['user_id'])  # Remove rows without valid user_id
df['user_id'] = df['user_id'].astype(int)

print(f"✓ Processing {len(df)} tweets from {df['user_id'].nunique()} unique users")

# ============================================================================
# 4. CHECK FOR EXISTING CHECKPOINT
# ============================================================================
tweet_results = []
start_idx = 0

if os.path.exists(CHECKPOINT_FILE):
    print(f"\n⚠ Found existing checkpoint: {CHECKPOINT_FILE}")
    response = input("Resume from checkpoint? (y/n): ").strip().lower()

    if response == 'y':
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)

        tweet_results = checkpoint['tweet_results']
        start_idx = checkpoint['last_index'] + 1

        print(f"✓ Resuming from tweet {start_idx}/{len(df)}")
        print(f"✓ Loaded {len(tweet_results)} previously processed tweets")
    else:
        print("Starting from scratch...")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

# ============================================================================
# 5. CLASSIFY EACH TWEET (with checkpoints)
# ============================================================================
print(f"\nClassifying tweets ({len(df) - start_idx} remaining)...")

for idx in tqdm(range(start_idx, len(df)), desc="Classifying tweets", initial=start_idx, total=len(df)):
    row = df.iloc[idx]

    # Get DNA for this tweet
    dna = get_agent_dna(row['text'])

    # Extract viewCount
    try:
        # Use regex to extract 'count': '<number>' from viewCount string
        # Note: count value is stored as a string, so we need to match the quotes
        match = re.search(r"'count':\s*'(\d+)'", str(row['viewCount']))
        view_count = int(match.group(1)) if match else 0
    except:
        view_count = 0

    # Store tweet-level results (convert to native Python types for JSON serialization)
    tweet_results.append({
        'user_id': int(row['user_id']),
        'tweet_id': int(row['id']) if pd.notna(row['id']) else 0,
        'view_count': int(view_count),
        'reply_count': int(row['replyCount']) if pd.notna(row['replyCount']) else 0,
        **dna
    })

    # Save checkpoint periodically
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_data = {
            'last_index': idx,
            'tweet_results': tweet_results,
            'total_tweets': len(df)
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f)

        tqdm.write(f"  ✓ Checkpoint saved at tweet {idx + 1}/{len(df)}")

# Final checkpoint save
checkpoint_data = {
    'last_index': len(df) - 1,
    'tweet_results': tweet_results,
    'total_tweets': len(df)
}
with open(CHECKPOINT_FILE, 'w') as f:
    json.dump(checkpoint_data, f)

print(f"\n✓ Classification complete!")

# ============================================================================
# 6. AGGREGATE BY USER
# ============================================================================
print("\nAggregating to user-level profiles...")

# Convert to DataFrame
tweets_df = pd.DataFrame(tweet_results)

def most_common(series):
    """Return the most common value in a series."""
    return series.mode()[0] if len(series.mode()) > 0 else series.iloc[0]

user_profiles = tweets_df.groupby('user_id').agg({
    # Count tweets per user
    'tweet_id': 'count',

    # Sum engagement metrics
    'view_count': 'sum',
    'reply_count': 'sum',

    # Average continuous scores
    'political_score': 'mean',
    'emotion_score': 'mean',
    'sentiment_score': 'mean',
    'hate_score': 'mean',
    'offensive_score': 'mean',

    # Most common labels (user's dominant political leaning/emotion/sentiment)
    'political_label': most_common,
    'emotion_label': most_common,
    'sentiment_label': most_common
}).reset_index()

# Rename tweet_id count to tweet_count
user_profiles.rename(columns={'tweet_id': 'tweet_count'}, inplace=True)

# Reorder columns for clarity
user_profiles = user_profiles[[
    'user_id', 'tweet_count', 'view_count', 'reply_count',
    'political_label', 'political_score',
    'emotion_label', 'emotion_score',
    'sentiment_label', 'sentiment_score',
    'hate_score', 'offensive_score'
]]

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
user_profiles.to_csv(OUTPUT_FILE, index=False)

# Clean up checkpoint file
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"✓ Removed checkpoint file")

print(f"\n{'='*80}")
print(f"CHUNK {CHUNK_NUM} PROCESSING COMPLETE!")
print(f"{'='*80}")
print(f"  Processed: {len(df)} tweets")
print(f"  Unique users: {len(user_profiles)}")
print(f"  Output: '{OUTPUT_FILE}'")
print(f"  Avg tweets/user: {user_profiles['tweet_count'].mean():.1f}")
print(f"  Total views: {user_profiles['view_count'].sum():,}")
print(f"  Avg views/user: {user_profiles['view_count'].mean():.1f}")
print(f"{'='*80}")
